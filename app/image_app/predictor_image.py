# app/image_app/predictor_image.py
"""
Lógica de inferencia — U-Net++ + Watershed
Carga el modelo una sola vez al iniciar la app (lifespan).
Recibe bytes de imagen, devuelve métricas + máscara en base64.
"""

import base64
import io
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp
from scipy import ndimage
from skimage.segmentation import watershed
from skimage.feature import peak_local_max

# ── Constantes de preprocesamiento (deben coincidir con entrenamiento) ────
IMG_SIZE        = (512, 512)
IMAGENET_MEAN   = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD    = np.array([0.229, 0.224, 0.225], dtype=np.float32)
THRESHOLD       = 0.5
WATERSHED_MIN_DISTANCE = 8

# ── Ruta del modelo ───────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent.parent   # raíz del proyecto
MODEL_PATH = BASE_DIR / "models" / "unetpp_best.pth"


# ─────────────────────────────────────────────────────────────────────────────
# Construcción del modelo (misma arquitectura que en entrenamiento)
# ─────────────────────────────────────────────────────────────────────────────

def build_unetpp() -> torch.nn.Module:
    return smp.UnetPlusPlus(
        encoder_name    = "efficientnet-b3",
        encoder_weights = None,          # pesos propios, no ImageNet
        in_channels     = 3,
        classes         = 1,
        activation      = None,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Singleton — el modelo se carga una sola vez
# ─────────────────────────────────────────────────────────────────────────────

_model  = None
_device = None


def load_model() -> None:
    """Llamar una sola vez desde el lifespan de FastAPI."""
    global _model, _device

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modelo no encontrado: {MODEL_PATH}\n"
            "Asegúrate de copiar unetpp_best.pth a models/"
        )

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[ImagePredictor] Cargando U-Net++ en {_device}...")

    _model = build_unetpp()
    state  = torch.load(MODEL_PATH, map_location=_device, weights_only=True)
    _model.load_state_dict(state)
    _model.to(_device)
    _model.eval()

    print(f"[ImagePredictor] Modelo listo — {MODEL_PATH.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Preprocesamiento (sin augmentation — modo inferencia)
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_bytes(image_bytes: bytes) -> torch.Tensor:
    """
    Convierte bytes de imagen (JPEG/PNG) al tensor listo para el modelo.
    Retorna tensor (1, 3, 512, 512) float32.
    """
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("No se pudo decodificar la imagen. Envía JPEG o PNG.")

    # BGR → RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize a 512×512 (igual que en entrenamiento)
    img = cv2.resize(img, (IMG_SIZE[1], IMG_SIZE[0]),
                     interpolation=cv2.INTER_LINEAR)

    # Normalización ImageNet (igual que normalize_imagenet en sartorius_utils)
    img = img.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD

    # HWC → CHW → batch
    tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)
    return tensor


# ─────────────────────────────────────────────────────────────────────────────
# Post-procesamiento — Watershed para contar instancias
# ─────────────────────────────────────────────────────────────────────────────

def apply_watershed(prob_map: np.ndarray,
                    threshold: float = THRESHOLD,
                    min_distance: int = WATERSHED_MIN_DISTANCE) -> tuple[np.ndarray, int]:
    """
    Aplica Watershed sobre el mapa de probabilidades.
    Retorna (máscara binaria uint8, número de células detectadas).
    """
    binary = (prob_map > threshold).astype(np.uint8)

    # Distance transform para encontrar centros
    dist   = ndimage.distance_transform_edt(binary)

    # Picos locales = centros de células
    coords = peak_local_max(dist, min_distance=min_distance, labels=binary)
    mask_peaks = np.zeros(dist.shape, dtype=bool)
    mask_peaks[tuple(coords.T)] = True

    markers, n_cells = ndimage.label(mask_peaks)

    # Watershed desde los centros
    labels = watershed(-dist, markers, mask=binary)

    final_mask = (labels > 0).astype(np.uint8) * 255
    return final_mask, n_cells


# ─────────────────────────────────────────────────────────────────────────────
# Métricas (sobre la propia predicción — sin ground truth en producción)
# ─────────────────────────────────────────────────────────────────────────────

def compute_self_metrics(prob_map: np.ndarray,
                         binary_mask: np.ndarray,
                         eps: float = 1e-6) -> dict[str, float]:
    """
    En producción no tenemos ground truth, así que reportamos métricas
    internas de la predicción:
    - confidence     : probabilidad media en la región segmentada
    - coverage       : fracción del frame cubierta por células
    - pred_dice      : Dice entre prob_map umbralizado y el mapa suavizado
                       (proxy de calidad de la segmentación)
    """
    pred_bin = (binary_mask > 0).astype(np.float32)
    prob_f   = prob_map.astype(np.float32)

    # Confianza media sobre píxeles positivos
    if pred_bin.sum() > 0:
        confidence = float(prob_f[pred_bin > 0].mean())
    else:
        confidence = 0.0

    # Cobertura del frame
    coverage = float(pred_bin.mean())

    # Dice entre binario duro y mapa de probs (proxy interno)
    inter  = (pred_bin * prob_f).sum()
    union  = pred_bin.sum() + prob_f.sum()
    dice   = float((2 * inter + eps) / (union + eps))

    # IoU binario vs suavizado
    iou    = float((inter + eps) / (union - inter + eps))

    return {
        "confidence" : round(confidence, 4),
        "coverage"   : round(coverage,   4),
        "dice_score" : round(dice,        4),
        "iou_score"  : round(iou,         4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Codificación de la máscara a base64 PNG
# ─────────────────────────────────────────────────────────────────────────────

def mask_to_base64(mask: np.ndarray) -> str:
    """
    Convierte máscara uint8 (H, W) a string base64 de PNG.
    Aplica colormap para que sea visualmente clara (verde sobre negro).
    """
    # Colormap: fondo negro, células en verde
    colored = np.zeros((*mask.shape, 3), dtype=np.uint8)
    colored[mask > 0] = [0, 255, 0]

    # Codificar como PNG en memoria
    _, buffer = cv2.imencode(".png", colored)
    b64 = base64.b64encode(buffer.tobytes()).decode("utf-8")
    return b64


# ─────────────────────────────────────────────────────────────────────────────
# Función principal de predicción
# ─────────────────────────────────────────────────────────────────────────────

def predict_image(image_bytes: bytes) -> dict:
    """
    Pipeline completo:
      bytes → preprocesar → U-Net++ → Watershed → métricas + máscara base64

    Retorna dict compatible con ImagePredictionResult (schemas_image.py).
    """
    if _model is None:
        raise RuntimeError("Modelo no cargado. Llama a load_model() primero.")

    # 1. Preprocesar
    tensor = preprocess_bytes(image_bytes).to(_device)

    # 2. Inferencia
    with torch.no_grad():
        logits   = _model(tensor)                        # (1, 1, 512, 512)
        prob_map = torch.sigmoid(logits).squeeze().cpu().numpy()  # (512, 512)

    # 3. Watershed
    binary_mask, cell_count = apply_watershed(prob_map)

    # 4. Métricas internas
    metrics = compute_self_metrics(prob_map, binary_mask)

    # 5. Máscara visual en base64
    mask_b64 = mask_to_base64(binary_mask)

    return {
        "cell_count"  : int(cell_count),
        "confidence"  : metrics["confidence"],
        "coverage"    : metrics["coverage"],
        "dice_score"  : metrics["dice_score"],
        "iou_score"   : metrics["iou_score"],
        "mask_base64" : mask_b64,
    }

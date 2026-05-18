# app/image_app/main_image.py
"""
API de segmentación de células — U-Net++ + Watershed
Puerto: 8001

Ejecutar:
    conda activate torch_img
    cd ~/DeepLearning/miniproyecto3
    uvicorn app.image_app.main_image:app --port 8001 --reload

Docs interactivos: http://localhost:8001/docs
"""

from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.image_app.predictor_image import load_model, predict_image, _device
from app.image_app.schemas_image import HealthImageResponse, ImagePredictionResult


# ─────────────────────────────────────────────────────────────────────────────
# Lifespan — carga el modelo una sola vez al arrancar
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield
    # cleanup si fuera necesario


# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "Sartorius Cell Segmentation API",
    description = (
        "Segmentación automática de células en imágenes microscópicas.\n\n"
        "**Modelo:** U-Net++ con encoder EfficientNet-B3 entrenado sobre "
        "el dataset Sartorius Cell Instance Segmentation (Kaggle).\n\n"
        "**Post-procesamiento:** Watershed para separar instancias individuales."
    ),
    version     = "1.0.0",
    lifespan    = lifespan,
)


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
def root():
    return {
        "message" : "Sartorius Cell Segmentation API",
        "docs"    : "http://localhost:8001/docs",
        "health"  : "http://localhost:8001/health",
    }


@app.get(
    "/health",
    response_model = HealthImageResponse,
    summary        = "Estado del servicio",
)
def health():
    from app.image_app.predictor_image import _model
    if _model is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")
    return HealthImageResponse(
        status   = "ok",
        model    = "U-Net++ EfficientNet-B3",
        device   = str(_device),
        img_size = "512x512",
    )


@app.post(
    "/predict_image",
    response_model = ImagePredictionResult,
    summary        = "Segmentar células en una imagen microscópica",
    description    = (
        "Recibe una imagen JPEG o PNG de microscopía celular y devuelve:\n"
        "- **cell_count**: número de células detectadas\n"
        "- **confidence**: confianza media del modelo sobre la región segmentada\n"
        "- **coverage**: fracción del frame cubierta por células\n"
        "- **dice_score**: métrica Dice interna\n"
        "- **iou_score**: métrica IoU interna\n"
        "- **mask_base64**: máscara PNG en base64 (células en verde sobre negro)\n\n"
        "La imagen se redimensiona internamente a 512×512 antes de la inferencia."
    ),
)
async def predict(
    file: UploadFile = File(..., description="Imagen JPEG o PNG de microscopía celular"),
):
    # Validar tipo de archivo
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(
            status_code = 422,
            detail      = f"Tipo de archivo no soportado: {file.content_type}. Usa JPEG o PNG.",
        )

    image_bytes = await file.read()

    if len(image_bytes) == 0:
        raise HTTPException(status_code=422, detail="El archivo está vacío.")

    try:
        result = predict_image(image_bytes)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en inferencia: {str(e)}")

    return ImagePredictionResult(**result)

# app/image_app/schemas_image.py
"""
Modelos Pydantic para la API de segmentación de imágenes.
"""

from pydantic import BaseModel, Field


class ImagePredictionResult(BaseModel):
    cell_count  : int   = Field(..., description="Número de células detectadas por Watershed")
    confidence  : float = Field(..., description="Probabilidad media sobre la región segmentada [0-1]")
    coverage    : float = Field(..., description="Fracción del frame cubierta por células [0-1]")
    dice_score  : float = Field(..., description="Dice interno pred_binario vs mapa de probabilidades")
    iou_score   : float = Field(..., description="IoU interno pred_binario vs mapa de probabilidades")
    mask_base64 : str   = Field(..., description="Máscara segmentada como PNG en base64 (células en verde)")

    model_config = {
        "json_schema_extra": {
            "example": {
                "cell_count"  : 38,
                "confidence"  : 0.8712,
                "coverage"    : 0.2341,
                "dice_score"  : 0.8530,
                "iou_score"   : 0.7431,
                "mask_base64" : "iVBORw0KGgoAAAANSUhEUgAA...",
            }
        }
    }


class HealthImageResponse(BaseModel):
    status      : str = Field(..., description="Estado del servicio")
    model       : str = Field(..., description="Modelo cargado")
    device      : str = Field(..., description="CPU o CUDA")
    img_size    : str = Field(..., description="Tamaño de entrada esperado")

    model_config = {
        "json_schema_extra": {
            "example": {
                "status"   : "ok",
                "model"    : "U-Net++ EfficientNet-B3",
                "device"   : "cuda",
                "img_size" : "512x512",
            }
        }
    }

# Deep Learning: Segmentación de Células e Interpretación de Texto

Proyecto integrador de deep learning en dos bloques: clasificación automática de currículos (NLP) y segmentación de células en imágenes microscópicas (Vision).

**Integrantes:** Mateo Gomez, Mateo Molinares, David Ibañez, Alejandro Moya  
**Jupyter Book:** https://undauphin.github.io/dl-resume-cell-classification/

---

## Resultados NLP

Dataset: Jarvis Calling Hiring Contest (Kaggle) — 2,484 currículos, 24 categorías profesionales.

| Modelo | Accuracy | F1 (weighted) | ROC-AUC |
|---|---|---|---|
| DistilBERT fine-tuned | **0.8016** | **0.7899** | 0.9746 |
| TF-IDF + XGBoost | 0.7748 | 0.7667 | **0.9808** |
| Word2Vec + BiLSTM | 0.7560 | 0.7396 | 0.9574 |
| CNN-1D | 0.6917 | 0.6589 | 0.9438 |
| FastText | 0.6461 | 0.6346 | 0.9473 |

**Modelo en producción:** TF-IDF + XGBoost — sin GPU, carga en menos de 1 segundo.

## Resultados Vision

Dataset: Sartorius Cell Instance Segmentation (Kaggle) — imágenes microscópicas.

| Modelo | Plataforma |
|---|---|
| U-Net++ + EfficientNet-B3 | Google Colab Pro (A100) |
| Cellpose 2.0 | Google Colab Pro |
| SAM / MedSAM | Google Colab Pro (A100) |
| HoVer-Net | Kaggle (P100) |
| Mask R-CNN + Swin | Kaggle (P100) |

**Modelo en producción:** U-Net++ — mejor IoU en test.

---

### Estructura

```
├── notebooks/
│   ├── nlp/          # 01 EDA → 07 Comparación
│   └── vision/       # 01 EDA → 07 Evaluación final
├── app/
│   ├── main.py           # API NLP (puerto 8000)
│   ├── predictor.py
│   ├── schemas.py
│   └── image_app/        # API Vision (puerto 8001)
├── docker/
│   ├── Dockerfile.nlp
│   └── Dockerfile.vision
├── docker-compose.yml
├── requirements.txt
└── requirements_image.txt
```

---

## Correr los notebooks

**NLP** (`conda activate tf_gpu`): ejecutar en orden del 01 al 07. El notebook 06 (DistilBERT) requiere al inicio:

```python
import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
```

**Vision** (`conda activate torch_img`): ejecutar en orden del 01 al 07.

---

## Correr las APIs

**API NLP** (puerto 8000):
```bash
conda activate tf_gpu
uvicorn app.main:app --reload
```

**API Vision** (puerto 8001):
```bash
conda activate torch_img
uvicorn app.image_app.main_image:app --port 8001 --reload
```

Documentación interactiva en `/docs` de cada puerto.

### Ejemplo NLP

```bash
curl -X POST "http://localhost:8000/predict_text" \
     -H "Content-Type: application/json" \
     -d '{"text": "Python developer with 8 years of experience..."}'
```

```json
{
  "category": "INFORMATION-TECHNOLOGY",
  "confidence": 0.9853,
  "top_3": [
    {"category": "INFORMATION-TECHNOLOGY", "confidence": 0.9853},
    {"category": "ENGINEERING", "confidence": 0.0014},
    {"category": "CONSULTANT", "confidence": 0.0013}
  ]
}
```

### Ejemplo Vision

```bash
curl -X POST "http://localhost:8001/predict_image" \
     -F "file=@imagen_celulas.png"
```

```json
{
  "cell_count": 38,
  "confidence": 0.8712,
  "coverage": 0.2341,
  "dice_score": 0.8530,
  "iou_score": 0.7431,
  "mask_base64": "iVBORw0KGgo..."
}
```

---

## Docker

Levanta ambas APIs con un solo comando:

```bash
docker compose up --build
```

- API NLP: `http://localhost:8000/docs`
- API Vision: `http://localhost:8001/docs`

> El servicio Vision requiere NVIDIA Container Toolkit para usar GPU. Sin él, corre en CPU.
# Deep Learning: Segmentación de Células e Interpretación de Texto

**Proyecto Integrador — Aprendizaje Automático**  
Universidad del Norte | Mayo 2026

**Integrantes:** Mateo Gomez, Mateo Molinares, David Ibañez, Alejandro Moya

---

## Descripción general

Este Jupyter Book documenta el desarrollo de un proyecto integrador de deep learning en dos bloques independientes pero complementarios.

### Bloque 1 — Segmentación de Imágenes

Segmentación automática de células en imágenes microscópicas usando el dataset **Sartorius Cell Instance Segmentation** (Kaggle). Se implementaron y compararon cinco arquitecturas: Cellpose 2.0, U-Net++ con encoder EfficientNet-B3, SAM/MedSAM, HoVer-Net y Mask R-CNN con backbone Swin. El modelo desplegado en producción es U-Net++, elegido por su mejor IoU en el conjunto de test.

### Bloque 2 — Procesamiento del Lenguaje Natural

Clasificación automática de currículums en 24 categorías profesionales usando el dataset **Resume Dataset** del Jarvis Calling Hiring Contest (Kaggle). Se implementaron cinco modelos: TF-IDF + XGBoost, CNN-1D, Word2Vec + BiLSTM, FastText nativo y DistilBERT fine-tuned. El mejor modelo en accuracy fue DistilBERT (80.16%), mientras que TF-IDF + XGBoost fue elegido para producción por su ligereza y tiempo de carga inferior a 1 segundo sin GPU.

---

## Estructura del proyecto

El proyecto incluye pipelines reproducibles, ajuste de hiperparámetros, evaluación con métricas estándar (Accuracy, F1, ROC-AUC) y despliegue como API dual con FastAPI y Docker Compose.

| Componente | Detalle |
|---|---|
| API NLP | FastAPI — puerto 8000 — modelo TF-IDF + XGBoost |
| API Vision | FastAPI — puerto 8001 — modelo U-Net++ |
| Despliegue | Docker Compose con dos contenedores independientes |

---

```{tableofcontents}
```

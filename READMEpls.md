# Miniproyecto NLP — Clasificación de Currículos

Clasificación automática de currículos en **24 categorías profesionales** usando modelos de deep learning y machine learning clásico.

**Autor:** Dr. Lihki Rubio  
**Dataset:** Jarvis Calling Hiring Contest (Kaggle) — 2,484 currículos

---

## Estructura

```
miniproyecto3/
├── data/
│   └── Resume.csv
├── notebooks/
│   ├── 01_EDA_NLP_Resume.ipynb
│   ├── 02_Model_TFIDF_XGBoost.ipynb
│   ├── 03_Model_CNN1D.ipynb
│   ├── 04_Model_BiLSTM.ipynb
│   ├── 05_Model_FastText.ipynb
│   ├── 06_Model_DistilBERT.ipynb
│   └── 07_Comparacion_Modelos.ipynb
├── app/
│   ├── main.py
│   ├── predictor.py
│   └── schemas.py
├── models/
├── artifacts/
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Resultados

| Modelo | Accuracy | F1 (weighted) | ROC-AUC |
|---|---|---|---|
| DistilBERT fine-tuned | **0.8016** | **0.7899** | 0.9746 |
| TF-IDF + XGBoost | 0.7748 | 0.7667 | **0.9808** |
| Word2Vec + BiLSTM | 0.7560 | 0.7396 | 0.9574 |
| CNN-1D | 0.6917 | 0.6589 | 0.9438 |
| FastText | 0.6461 | 0.6346 | 0.9473 |

**Mejor modelo general:** DistilBERT (mayor accuracy y F1)  
**Mejor modelo liviano:** TF-IDF + XGBoost (segunda posición, sin GPU, desplegado en la API)

---

## Correr los notebooks

Activar el entorno y abrir Jupyter:

```bash
conda activate tf_gpu
jupyter notebook
```

Ejecutar en orden del 01 al 07. El notebook 06 (DistilBERT) requiere una celda adicional al inicio:

```python
import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
```

---

## Correr la API

Desde la raíz del proyecto:

```bash
conda activate tf_gpu
uvicorn app.main:app --reload
```

La API queda disponible en `http://localhost:8000`.  
Documentación interactiva en `http://localhost:8000/docs`.

### Endpoints

| Método | Ruta | Descripción |
|---|---|---|
| GET | `/health` | Estado de la API y modelo cargado |
| POST | `/predict_text` | Clasifica un currículo |

### Ejemplo de uso

```bash
curl -X POST "http://localhost:8000/predict_text" \
     -H "Content-Type: application/json" \
     -d '{"text": "Python developer with 8 years of experience..."}'
```

Respuesta:

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

---

## Docker

```bash
docker build -t nlp-resume-api .
docker run -p 8000:8000 nlp-resume-api
```

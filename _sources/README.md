# Miniproyecto NLP — Clasificación Automática de Currículos

**Autores:** Alejandro Moya, Mateo Gomez, Mateo Molinares, David Ibañez
**Entorno:** Python 3.10 · WSL2 Ubuntu · Conda `tf_gpu` · GPU RTX 4050 Laptop (6 GB VRAM)

Proyecto integrador de aprendizaje automático para la **clasificación automática de currículos en 24 categorías profesionales**, usando el dataset [Jarvis Calling Hiring Contest](https://www.kaggle.com/) de Kaggle. Incluye pipeline reproducible, comparación de cinco modelos de NLP y despliegue básico como API REST.

---

## Tabla de contenidos

1. [Dataset](#dataset)
2. [Resultados comparativos](#resultados-comparativos)
3. [Estructura del proyecto](#estructura-del-proyecto)
4. [Instalación y reproducción](#instalación-y-reproducción)
5. [Correr la API](#correr-la-api)
6. [Decisiones técnicas importantes](#decisiones-técnicas-importantes)
7. [MLOps — Docker](#mlops--docker)
8. [Monitoreo conceptual](#monitoreo-conceptual)

---

## Dataset

- **Archivo:** `data/Resume.csv`
- **Dimensiones:** 2,484 currículos × 4 columnas
- **Columna de entrada:** `Resume_str` (texto crudo del CV)
- **Columna objetivo:** `Category` (24 clases profesionales)
- **Desbalance:** ratio máx/mín de 5.45x (120 muestras en IT vs 22 en BPO)

### Clases (24)

`ACCOUNTANT` · `ADVOCATE` · `AGRICULTURE` · `APPAREL` · `ARTS` · `AUTOMOBILE` · `AVIATION` · `BANKING` · `BPO` · `BUSINESS-DEVELOPMENT` · `CHEF` · `CONSTRUCTION` · `CONSULTANT` · `DESIGNER` · `DIGITAL-MEDIA` · `ENGINEERING` · `FINANCE` · `FITNESS` · `HEALTHCARE` · `HR` · `INFORMATION-TECHNOLOGY` · `PUBLIC-RELATIONS` · `SALES` · `TEACHER`

### Split

| Conjunto | Proporción | Muestras |
|---|---|---|
| Train | 70% | 1,738 |
| Validation | 15% | 373 |
| Test | 15% | 373 |

Split estratificado con `random_state=42`. El tokenizador/vectorizador se fittea **solo sobre train** para evitar data leakage.

---

## Resultados comparativos

Todas las métricas reportadas sobre el conjunto **test** (visto una sola vez). Promedio `weighted` por el desbalance de clases.

| Modelo | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| TF-IDF + XGBoost | 0.7748 | 0.7816 | 0.7748 | 0.7667 | 0.9808 |
| CNN-1D | 0.6917 | 0.6582 | 0.6917 | 0.6589 | 0.9438 |
| Word2Vec + BiLSTM | 0.7560 | 0.7427 | 0.7560 | 0.7396 | 0.9574 |
| FastText | 0.6461 | 0.6399 | 0.6461 | 0.6346 | 0.9473 |
| **DistilBERT fine-tuned** | **0.8016** | **0.7850** | **0.8016** | **0.7899** | **0.9746** |

**Mejor modelo:** DistilBERT fine-tuned (F1 = 0.7899, ROC-AUC = 0.9746).  
**Mejor modelo clásico:** TF-IDF + XGBoost — solo 0.023 de F1 por debajo de DistilBERT, con una fracción del costo computacional.  
**Modelo en producción (API):** TF-IDF + XGBoost — liviano, sin dependencia de GPU, carga en < 1 segundo.

---

## Estructura del proyecto

```
miniproyecto3/
├── data/
│   └── Resume.csv                    # Dataset original
│
├── notebook/
│   ├── 01_EDA_NLP_Resume.ipynb       # Análisis exploratorio de datos
│   ├── 02_Model_TFIDF_XGBoost.ipynb  # Baseline: TF-IDF + XGBoost
│   ├── 03_Model_CNN1D.ipynb          # CNN-1D con embeddings entrenables
│   ├── 04_Model_BiLSTM.ipynb         # Word2Vec + BiLSTM + Keras Tuner
│   ├── 05_Model_FastText.ipynb       # FastText nativo (fasttext-numpy2)
│   ├── 06_Model_DistilBERT.ipynb     # DistilBERT fine-tuning (TF)
│   └── 07_Comparacion_Modelos.ipynb  # Tabla comparativa y análisis crítico
│
├── artifacts/
│   ├── tokenizer.pkl                 # Keras Tokenizer (fit solo sobre train)
│   ├── label_encoder.pkl             # LabelEncoder — 24 clases
│   ├── config.pkl                    # MAX_LEN=100, VOCAB_SIZE=20000, NUM_CLASSES=24
│   ├── X_train.npy / X_val.npy / X_test.npy   # Secuencias padded
│   ├── y_train.npy / y_val.npy / y_test.npy   # Etiquetas codificadas
│   ├── fasttext_tmp/                 # Archivos .txt para FastText
│   └── metrics/                      # JSON de métricas por modelo
│       ├── tfidf_xgboost.json
│       ├── cnn1d_metrics.json
│       ├── bilstm_metrics.json
│       ├── fasttext_metrics.json
│       └── distilbert_metrics.json
│
├── models/
│   ├── tfidf_xgboost.json            # Modelo XGBoost (formato nativo)
│   ├── tfidf_xgboost_tfidf.pkl       # Vectorizador TF-IDF
│   ├── cnn1d_model.keras
│   ├── bilstm_w2v.keras
│   ├── fasttext_model.bin
│   └── distilbert_classifier.keras
│
├── app/
│   ├── main.py                       # Servidor FastAPI
│   ├── predictor.py                  # Carga de modelos y lógica de predicción
│   └── schemas.py                    # Modelos Pydantic (input/output)
│
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Instalación y reproducción

### Requisitos

- Miniconda o Anaconda
- Python 3.10
- GPU con CUDA (opcional — los notebooks de Keras y XGBoost lo detectan automáticamente)

### Crear el entorno

```bash
conda create -n tf_gpu python=3.10
conda activate tf_gpu
pip install tensorflow[and-cuda] xgboost scikit-learn gensim nltk \
    fasttext-numpy2 transformers==4.35.2 tokenizers==0.15.2 \
    huggingface_hub keras-tuner --no-deps \
    matplotlib seaborn wordcloud pandas numpy jupyter
```

### Correr los notebooks en orden

```bash
cd miniproyecto3/
jupyter notebook
```

Ejecutar en este orden:

| # | Notebook | Descripción |
|---|---|---|
| 01 | `01_EDA_NLP_Resume.ipynb` | EDA — genera artefactos en `artifacts/` |
| 02 | `02_Model_TFIDF_XGBoost.ipynb` | Baseline — genera `models/tfidf_xgboost.*` |
| 03 | `03_Model_CNN1D.ipynb` | CNN-1D |
| 04 | `04_Model_BiLSTM.ipynb` | BiLSTM con Keras Tuner |
| 05 | `05_Model_FastText.ipynb` | FastText nativo |
| 06 | `06_Model_DistilBERT.ipynb` | DistilBERT fine-tuning |
| 07 | `07_Comparacion_Modelos.ipynb` | Comparación final y análisis crítico |

> **Importante:** El notebook 06 requiere `TF_USE_LEGACY_KERAS=1` como primera línea absoluta antes de cualquier import. Los demás notebooks no deben setearlo.

---

## Correr la API

La API usa el modelo **TF-IDF + XGBoost** — sin GPU, sin TensorFlow, carga en < 1 segundo.

### Localmente

```bash
conda activate tf_gpu
cd miniproyecto3/
uvicorn app.main:app --reload
```

La API queda disponible en `http://localhost:8000`.  
Documentación interactiva (Swagger UI): `http://localhost:8000/docs`

### Endpoints

| Método | Ruta | Descripción |
|---|---|---|
| GET | `/` | Información básica de la API |
| GET | `/health` | Estado del servidor y modelo cargado |
| POST | `/predict_text` | Clasificar un currículo |

### Ejemplo de uso

**Request:**
```bash
curl -X POST "http://localhost:8000/predict_text" \
     -H "Content-Type: application/json" \
     -d '{"text": "Python developer with 8 years of experience. Proficient in Django, REST APIs, PostgreSQL and AWS. Led development of microservices architecture."}'
```

**Response:**
```json
{
  "category": "INFORMATION-TECHNOLOGY",
  "confidence": 0.9423,
  "top_3": [
    {"category": "INFORMATION-TECHNOLOGY", "confidence": 0.9423},
    {"category": "ENGINEERING", "confidence": 0.0312},
    {"category": "CONSULTANT", "confidence": 0.0089}
  ]
}
```

> **Nota:** El modelo fue entrenado con CVs completos (promedio 584 palabras limpias). Con textos muy cortos o genéricos, la confianza cae y las predicciones son menos confiables — comportamiento esperado en TF-IDF sin contexto semántico.

---

## Decisiones técnicas importantes

### Data leakage — tokenizador y vectorizador

El `Tokenizer` de Keras y el `TfidfVectorizer` se fittean **solo sobre el conjunto train**, luego se aplican a val y test. El split de índices se hace primero, el fit después. El `LabelEncoder` sí puede fitearse sobre todas las etiquetas porque solo aprende nombres de clases.

### MAX_LEN = 100 para modelos Keras

Los textos limpios promedian 584 palabras con un P95 de 1,008. Se seleccionó MAX_LEN=100 por restricciones de VRAM (RTX 4050, 6 GB). Aumentar a 300-500 habría incrementado la capacidad del modelo sin garantía de mejor generalización — el cuello de botella es el volumen de datos (1,738 muestras de train), no la longitud del contexto.

### XGBoost — subsample y colsample_bytree fijados en 1.0

Con matrices TF-IDF sparse de alta dimensionalidad (50,000 features), reducir `subsample` o `colsample_bytree` destabiliza el gradiente. La regularización se maneja vía L1/L2 (`reg_alpha=1.0`, `reg_lambda=5.0`).

### DistilBERT — carga de pesos

Las clases `TF*` de HuggingFace son incompatibles con Keras 3.x. Se usa `TFDistilBertModel` (backbone directo) con pesos cargados desde `tf_model.h5` descargado con `hf_hub_download`, evitando la conversión PyTorch → TF de transformers. Requiere `transformers==4.35.2`.

### Desbalance de clases

Ratio máx/mín de 5.45x. Se descartó `class_weight` en Keras y `sample_weight` en XGBoost por inestabilidad del entrenamiento. Todas las métricas se reportan en modo `weighted` para evaluación justa entre clases mayoritarias y minoritarias.

---

## MLOps — Docker

La API está containerizada. Para construir y correr el contenedor:

```bash
# Construir la imagen
docker build -t nlp-resume-api .

# Correr el contenedor
docker run -p 8000:8000 nlp-resume-api
```

El `Dockerfile` usa `python:3.10-slim` e incluye solo las dependencias necesarias para la API (sin TensorFlow, sin librerías de notebooks). Los modelos y artefactos se copian al contenedor en tiempo de build.

---

## Monitoreo conceptual

En un entorno de producción real, se monitorearían los siguientes aspectos:

**Data drift:** comparar la distribución del vocabulario de los textos entrantes con la distribución del set de entrenamiento. Un aumento en tokens OOV (fuera del vocabulario TF-IDF) o cambios en la longitud promedio de los textos pueden indicar drift. Herramientas: Evidently AI, WhyLogs.

**Métricas en producción:** registrar la confianza promedio por predicción y la distribución de categorías predichas. Una caída sostenida en la confianza o una distribución muy diferente a la del test pueden indicar degradación del modelo.

**Latencia:** monitorear el tiempo de respuesta del endpoint `/predict_text`. Con TF-IDF + XGBoost se espera latencia < 100ms por request en CPU. Alertar si supera 500ms sostenidamente.

**Reentrenamiento:** evaluar reentrenamiento periódico si se acumulan suficientes ejemplos con etiquetas corregidas por humanos o si las métricas en producción se degradan más del 5% respecto al baseline de test.

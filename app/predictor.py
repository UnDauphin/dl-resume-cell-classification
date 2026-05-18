import os
import re
import pickle
import numpy as np
import xgboost as xgb
import nltk

nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

# ── Rutas ─────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODELS_DIR    = os.path.join(BASE_DIR, 'models')
ARTIFACTS_DIR = os.path.join(BASE_DIR, 'artifacts')

TFIDF_PATH   = os.path.join(MODELS_DIR,    'tfidf_xgboost_tfidf.pkl')
MODEL_PATH   = os.path.join(MODELS_DIR,    'tfidf_xgboost.json')
ENCODER_PATH = os.path.join(ARTIFACTS_DIR, 'label_encoder.pkl')

# ── Carga de artefactos (una sola vez al iniciar la API) ──────────────────────
STOPWORDS = set(stopwords.words('english'))

with open(TFIDF_PATH, 'rb') as f:
    tfidf = pickle.load(f)

with open(ENCODER_PATH, 'rb') as f:
    le = pickle.load(f)

model = xgb.XGBClassifier()
model.load_model(MODEL_PATH)

CLASS_NAMES = le.classes_

# ── Preprocesamiento (idéntico al notebook 02) ────────────────────────────────
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'http\S+|www\.\S+|\S+@\S+', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = [w for w in text.split() if w not in STOPWORDS and len(w) >= 3]
    return ' '.join(tokens)

# ── Predicción ────────────────────────────────────────────────────────────────
def predict_resume(text: str) -> dict:
    cleaned   = clean_text(text)
    vectorized = tfidf.transform([cleaned])
    proba     = model.predict_proba(vectorized)[0]

    top3_idx  = np.argsort(proba)[::-1][:3]
    top3      = [
        {"category": CLASS_NAMES[i], "confidence": round(float(proba[i]), 4)}
        for i in top3_idx
    ]

    return {
        "category"  : CLASS_NAMES[top3_idx[0]],
        "confidence": round(float(proba[top3_idx[0]]), 4),
        "top_3"     : top3,
    }
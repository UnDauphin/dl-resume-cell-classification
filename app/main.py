from fastapi import FastAPI, HTTPException
from app.schemas import ResumeInput, PredictionResult, HealthResponse
from app.predictor import predict_resume, CLASS_NAMES

app = FastAPI(
    title="NLP Resume Classifier API",
    description="Clasificación automática de currículos en 24 categorías profesionales. Modelo: TF-IDF + XGBoost.",
    version="1.0.0",
)

# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
def health():
    return {
        "status" : "ok",
        "model"  : "TF-IDF + XGBoost",
        "classes": len(CLASS_NAMES),
    }

# ── Predicción de texto ───────────────────────────────────────────────────────
@app.post("/predict_text", response_model=PredictionResult)
def predict_text(input: ResumeInput):
    try:
        result = predict_resume(input.text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ── Root ──────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "message"  : "Resume Classifier API — usa /predict_text para predecir",
        "docs"     : "/docs",
        "health"   : "/health",
    }
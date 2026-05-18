from pydantic import BaseModel, Field
from typing import List

class ResumeInput(BaseModel):
    text: str = Field(
        ...,
        min_length=10,
        description="Texto crudo del currículo a clasificar"
    )

class PredictionResult(BaseModel):
    category: str
    confidence: float
    top_3: List[dict]

class HealthResponse(BaseModel):
    status: str
    model: str
    classes: int
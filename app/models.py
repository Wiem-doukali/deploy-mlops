# app/models.py
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime

class DiagnosisRequest(BaseModel):
    symptoms_text: str
    patient_age: Optional[int] = None
    patient_gender: Optional[str] = None

class DiagnosisResponse(BaseModel):
    consultation_id: int
    symptoms: List[str]
    predictions: List[Dict[str, Any]]
    recommendations: str
    severity: str
    timestamp: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    diseases_count: int
    whisper_available: bool
    modules_loaded: bool
    timestamp: str
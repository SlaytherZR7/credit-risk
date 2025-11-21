from typing import List, Dict, Any
from pydantic import BaseModel

class PredictionRequest(BaseModel):
    features: List[Dict]   # ← acepta todo, sin validación de tipos

class BatchPredictionRequest(BaseModel):
    features: List[Dict[str, Any]]

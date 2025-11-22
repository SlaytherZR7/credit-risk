from typing import List, Dict
from pydantic import BaseModel

class PredictionRequest(BaseModel):
    features: List[Dict]

class BatchPredictionRequest(BaseModel):
    features: List[Dict]

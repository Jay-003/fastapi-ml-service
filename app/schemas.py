from typing import List
from pydantic import BaseModel, conlist


class PredictRequest(BaseModel):
    # enforce at least one feature and floats
    features: conlist(float, min_items=1)


class PredictResponse(BaseModel):
    prediction: int
    confidence: float
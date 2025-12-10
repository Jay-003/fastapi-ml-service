from pydantic import BaseModel
from typing import List

class PredictRequest(BaseModel):
    features: List[float]

class SimpleModel:
    """A tiny deterministic "model" packaged with the app.
    Replace this with scikit-learn / joblib model load when needed.
    """
    def __init__(self):
        # example: pretend coefficients
        self.coefs = [0.4, -0.2, 0.1, 0.05]

    def predict(self, features: List[float]):
        # simple dot-product + threshold
        s = 0.0
        for i, v in enumerate(features):
            if i < len(self.coefs):
                s += self.coefs[i] * v
            else:
                s += 0.01 * v
        return 1 if s > 0.5 else 0

import logging
from typing import Sequence, Tuple, Any
import numpy as np

logger = logging.getLogger(__name__)


def predict(model: Any, features: Sequence[float]) -> Tuple[int, float]:
    """
    Run prediction using the provided model and features.
    Returns (predicted_class, confidence) where confidence is between 0 and 1.
    """
    X = np.array(features, dtype=float).reshape(1, -1)
    logger.debug("Running prediction for input: %s", X)

    # Try predict_proba first (for classifiers)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[0]
        pred_idx = int(probs.argmax())
        confidence = float(probs[pred_idx])
        logger.debug("predict_proba used. pred=%s, conf=%s", pred_idx, confidence)
        return pred_idx, confidence

    # Fallback to decision_function if available (scale to probabilities using sigmoid)
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        # If binary classification decision_function returns shape (n_samples,), else per-class
        try:
            # multiclass
            probs = _softmax(scores[0])
            pred_idx = int(np.argmax(probs))
            confidence = float(probs[pred_idx])
        except Exception:
            # binary
            score = float(scores[0])
            confidence = _sigmoid(score)
            pred_idx = int(model.predict(X)[0])
        logger.debug("decision_function used. pred=%s, conf=%s", pred_idx, confidence)
        return pred_idx, confidence

    # Last resort: use predict and set confidence 1.0 (no probability info)
    pred = int(model.predict(X)[0])
    logger.warning("Model has no probability or decision function; returning confidence=1.0")
    return pred, 1.0


def _sigmoid(x: float) -> float:
    import math

    return 1 / (1 + math.exp(-x))


def _softmax(scores):
    exp = np.exp(scores - np.max(scores))
    return exp / exp.sum(axis=-1, keepdims=True)
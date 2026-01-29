import logging
from typing import Any
import joblib
from pathlib import Path


logger = logging.getLogger(__name__)


def load_model(path: str) -> Any:
    """
    Load and return the model saved with joblib.
    Raises FileNotFoundError or joblib-specific errors if loading fails.
    """
    model_path = Path(path)
    if not model_path.exists():
        logger.error("Model file not found at %s", path)
        raise FileNotFoundError(f"Model file not found at {path}")
    logger.info("Loading model from %s", path)
    model = joblib.load(path)
    logger.info("Model loaded successfully")
    return model
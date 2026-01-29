import os
import sys
import subprocess
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# make sure app module is importable
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.main import app  # noqa: E402


MODEL_PATH = ROOT / "models" / "model.joblib"
TRAIN_SCRIPT = ROOT / "scripts" / "train_model.py"


@pytest.fixture(scope="session", autouse=True)
def ensure_model():
    # If model doesn't exist, train it
    if not MODEL_PATH.exists():
        subprocess.check_call([sys.executable, str(TRAIN_SCRIPT)])
    # set environment so app uses the correct model path when imported
    os.environ["MODEL_PATH"] = str(MODEL_PATH)
    return str(MODEL_PATH)


def test_health(ensure_model):
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True


def test_predict_valid(ensure_model):
    client = TestClient(app)
    payload = {"features": [5.1, 3.5, 1.4, 0.2]}
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert "prediction" in data
    assert "confidence" in data
    assert isinstance(data["prediction"], int)
    assert 0.0 <= float(data["confidence"]) <= 1.0


def test_predict_invalid_payload(ensure_model):
    client = TestClient(app)
    # empty features should be rejected
    payload = {"features": []}
    r = client.post("/predict", json=payload)
    assert r.status_code == 422 or r.status_code == 400
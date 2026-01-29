import os
import logging
from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.status import HTTP_500_INTERNAL_SERVER_ERROR
from pydantic import ValidationError
from dotenv import load_dotenv

from .schemas import PredictRequest, PredictResponse
from .model_loader import load_model
from .services.prediction_service import predict

# load environment variables from .env (if present)
load_dotenv()

# configuration via environment
MODEL_PATH = os.getenv("MODEL_PATH", "models/model.joblib")
PORT = int(os.getenv("PORT", "8000"))

# logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger("fastapi-ml-service")

app = FastAPI(title="fastapi-ml-service", version="0.1.0")

# Allow CORS for local testing; adjust origins in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event():
    try:
        app.state.model = load_model(MODEL_PATH)
        logger.info("Model loaded and ready")
    except Exception as exc:
        # Keep app running but record model load failure
        app.state.model = None
        logger.exception("Failed to load model on startup: %s", exc)


def get_model(request: Request):
    model = getattr(request.app.state, "model", None)
    if model is None:
        logger.error("Model is not loaded; cannot serve predictions")
        raise HTTPException(status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail="Model not loaded")
    return model


@app.get("/health")
def health():
    model_loaded = getattr(app.state, "model", None) is not None
    return {"status": "ok", "model_loaded": model_loaded}


@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(payload: PredictRequest, model=Depends(get_model)):
    try:
        pred, conf = predict(model, payload.features)
        return PredictResponse(prediction=pred, confidence=round(conf, 4))
    except ValidationError as ve:
        logger.exception("Validation error during prediction: %s", ve)
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as exc:
        logger.exception("Unexpected error during prediction: %s", exc)
        raise HTTPException(status_code=500, detail="Prediction failed")


@app.exception_handler(HTTPException)
def http_exception_handler(request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.exception_handler(Exception)
def generic_exception_handler(request, exc: Exception):
    logger.exception("Unhandled exception: %s", exc)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=PORT, reload=False)
## Features

- FastAPI + Uvicorn
- Pydantic request validation
- Simple scikit-learn model (RandomForest) saved with joblib
- Model loaded on startup
- Endpoints:
  - `GET /health` - service health & model status
  - `POST /predict` - accepts features and returns prediction with confidence
- Tests with pytest and TestClient
- Docker support
- Environment configuration via `.env`
- Logging and error handling

## How to run (dev)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

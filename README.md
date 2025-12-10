# Python API + ML Demo (FastAPI)

**Stack:** Python 3.10+, FastAPI, Uvicorn, SQLite (or any DB), SQLAlchemy (optional)

## What this contains
- `app/main.py` — FastAPI app with example endpoints:
  - `/health` — health check
  - `/predict` — sample POST endpoint that returns predictions from a simple model
  - `/items` — simple CRUD-ish endpoint using an in-memory store
- `app/model.py` — simple placeholder "model" class (self-contained; no heavy deps) and example of how to wrap a scikit-learn model if desired.
- `requirements.txt` — packages to install
- `Dockerfile` — containerize the service
- `README.md` — this file
- `resume_bullets.md` — ready-to-paste resume bullets for this project

## How to run (dev)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

## Resume bullets
See `resume_bullets.md`.

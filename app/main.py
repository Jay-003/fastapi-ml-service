from fastapi import FastAPI
from pydantic import BaseModel

from .model import SimpleModel, PredictRequest

app = FastAPI(title="Python API + ML Demo")
model = SimpleModel()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(req: PredictRequest):
    # returns a simple deterministic "prediction" without external dependencies
    pred = model.predict(req.features)
    return {"prediction": pred}

items = {}

class Item(BaseModel):
    id: int
    name: str
    price: float

@app.post("/items")
def create_item(item: Item):
    items[item.id] = item.dict()
    return items[item.id]

@app.get("/items/{item_id}")
def get_item(item_id: int):
    return items.get(item_id, {"error": "not found"})

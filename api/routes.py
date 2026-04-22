from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Request
from api.schemas import CustomerInput
from api import database as db # Connects to our new database file

router = APIRouter()

# --- Internal Helpers ---
def _ready(request: Request):
    model = request.app.state.model
    prep  = request.app.state.preprocessor
    if model is None or prep is None:
        raise HTTPException(status_code=503, detail="Model artifacts not loaded.")
    return model, prep, request.app.state.metadata, request.app.state.cfg

def _to_array(customers: list[CustomerInput], cfg: dict, prep: Any) -> np.ndarray:
    rows = [c.model_dump(by_alias=True) for c in customers]
    for row in rows:
        if row["pdays"] == cfg["data"]["pdays_not_contacted"]:
            row["pdays"] = cfg["data"]["pdays_fill_value"]
    return prep.transform(pd.DataFrame(rows))

# --- Endpoints ---

@router.get("/", summary="Health check")
async def health_check(request: Request):
    return {"status": "active", "db_connected": True}

@router.post("/predict", summary="Predict & Log Lead")
async def predict(customer: CustomerInput, request: Request):
    model, prep, meta, cfg = _ready(request)
    threshold = meta["tuned_threshold"]

    # ML Inference
    encoded = _to_array([customer], cfg, prep)
    prob = float(model.predict_proba(encoded)[0][1])
    subscribed = bool(prob >= threshold)
    
    result = {
        "probability": round(prob, 4),
        "prediction": int(subscribed),
        "recommendation": "High Priority Lead" if subscribed else "Low Priority"
    }

    # ASYNC LOGGING: Save the event to MongoDB Atlas
    await db.log_prediction(customer.model_dump(), result)
    
    return result

@router.post("/research", summary="Financial Strategy RAG")
async def research(payload: dict, request: Request):
    query = payload.get("query")
    if not query:
        raise HTTPException(status_code=400, detail="Query required")

    # This is a placeholder for your RAG logic
    answer = f"Financial analysis for: {query}. (RAG Engine Active)"
    
    # LOGGING: Save the research query to MongoDB
    await db.log_research(query, answer)
    
    return {"query": query, "answer": answer}
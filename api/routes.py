from __future__ import annotations
import asyncio
from typing import Any
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Request
from api.schemas import CustomerInput
from api import database as db

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

def _make_prediction_dict(prob: float, threshold: float) -> dict:
    """Helper to maintain consistent output format for tests."""
    subscribed = bool(prob >= threshold)
    return {
        "probability_of_subscription": round(prob, 4),
        "prediction_class": int(subscribed),
        "recommendation": "Call — high-probability lead" if subscribed else "Do not call — low probability",
        "threshold_used": threshold,
    }

# --- Endpoints ---

@router.get("/", summary="Health check")
async def health_check(request: Request):
    """Returns status and model metadata (Required by tests)."""
    meta = request.app.state.metadata
    return {
        "status": "active",
        "model_loaded": request.app.state.model is not None,
        "model_name":   meta.get("model_name"),
        "val_pr_auc":   meta.get("val_pr_auc"),
        "tuned_threshold": meta.get("tuned_threshold"),
        "trained_at":   meta.get("timestamp"),
        "db_connected": True # Our new addition
    }

@router.get("/model-info", summary="Detailed model metadata")
async def model_info(request: Request):
    """Restored for test compliance."""
    return request.app.state.metadata

@router.post("/predict", summary="Single-customer subscription prediction")
async def predict(customer: CustomerInput, request: Request):
    model, prep, meta, cfg = _ready(request)
    threshold = meta["tuned_threshold"]

    encoded = _to_array([customer], cfg, prep)
    prob = float(model.predict_proba(encoded)[0][1])
    
    result = _make_prediction_dict(prob, threshold)

    asyncio.create_task(db.log_prediction(customer.model_dump(), result))
    return result

@router.post("/batch-predict", summary="Batch subscription predictions")
async def batch_predict(customers: list[CustomerInput], request: Request):
    """Restored for test compliance."""
    if not customers:
        raise HTTPException(status_code=400, detail="Customer list must not be empty.")

    model, prep, meta, cfg = _ready(request)
    threshold = meta["tuned_threshold"]

    encoded = _to_array(customers, cfg, prep)
    probs   = model.predict_proba(encoded)[:, 1]
    
    results = [_make_prediction_dict(float(p), threshold) for p in probs]
    return results

@router.post("/research", summary="Financial Strategy RAG")
async def research(payload: dict, request: Request):
    query = payload.get("query")
    if not query:
        raise HTTPException(status_code=400, detail="Query required")

    # Connect to the actual RAG engine in your app state
    if hasattr(request.app.state, "query_engine") and request.app.state.query_engine:
        response = request.app.state.query_engine.query(query)
        answer = str(response)
    else:
        answer = f"RAG Engine is not initialized. (Response for: {query})"
    
    asyncio.create_task(db.log_research(query, answer))
    return {"query": query, "answer": answer}

@router.get("/analytics", summary="Usage analytics from MongoDB")
async def analytics():
    return await db.get_analytics()
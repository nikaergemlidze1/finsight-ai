from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Request

from api.schemas import CustomerInput

router = APIRouter()


def _ready(request: Request) -> tuple[Any, Any, dict, dict]:
    """Return (model, preprocessor, metadata, cfg) or raise 503."""
    model = request.app.state.model
    prep  = request.app.state.preprocessor
    if model is None or prep is None:
        raise HTTPException(status_code=503, detail="Model artifacts not loaded. Run the training pipeline first.")
    return model, prep, request.app.state.metadata, request.app.state.cfg


def _to_array(customers: list[CustomerInput], cfg: dict, prep: Any) -> np.ndarray:
    """Convert a list of CustomerInput → preprocessed numpy array."""
    pdays_raw  = cfg["data"]["pdays_not_contacted"]
    pdays_fill = cfg["data"]["pdays_fill_value"]

    rows = []
    for c in customers:
        row = c.model_dump(by_alias=True)   # keeps emp.var.rate, cons.price.idx, etc.
        if row["pdays"] == pdays_raw:
            row["pdays"] = pdays_fill       # mirror the recode from data_processing.py
        rows.append(row)

    df = pd.DataFrame(rows)
    return prep.transform(df)


def _make_prediction(prob: float, threshold: float) -> dict:
    subscribed = bool(prob >= threshold)
    return {
        "probability_of_subscription": round(prob, 4),
        "prediction_class": int(subscribed),
        "recommendation": "Call — high-probability lead" if subscribed else "Do not call — low probability",
        "threshold_used": threshold,
    }


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/", summary="Health check")
def health_check(request: Request) -> dict:
    meta = request.app.state.metadata
    return {
        "status": "active",
        "model_loaded": request.app.state.model is not None,
        "model_name":   meta.get("model_name"),
        "val_pr_auc":   meta.get("val_pr_auc"),
        "tuned_threshold": meta.get("tuned_threshold"),
        "trained_at":   meta.get("timestamp"),
    }


@router.get("/model-info", summary="Detailed model metadata")
def model_info(request: Request) -> dict:
    return request.app.state.metadata


@router.post("/predict", summary="Single-customer subscription prediction")
def predict(customer: CustomerInput, request: Request) -> dict:
    model, prep, meta, cfg = _ready(request)
    threshold = meta["tuned_threshold"]

    encoded = _to_array([customer], cfg, prep)
    prob = float(model.predict_proba(encoded)[0][1])
    return _make_prediction(prob, threshold)


@router.post("/batch-predict", summary="Batch subscription predictions")
def batch_predict(customers: list[CustomerInput], request: Request) -> list[dict]:
    if not customers:
        raise HTTPException(status_code=400, detail="Customer list must not be empty.")

    model, prep, meta, cfg = _ready(request)
    threshold = meta["tuned_threshold"]

    encoded = _to_array(customers, cfg, prep)
    probs   = model.predict_proba(encoded)[:, 1]
    return [_make_prediction(float(p), threshold) for p in probs]

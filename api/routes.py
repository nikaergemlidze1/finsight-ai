from __future__ import annotations

import asyncio
import logging
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request

from api import database as db
from api.rate_limit import limit_analytics, limit_batch, limit_predict, limit_research
from api.schemas import CustomerInput, ResearchQuery
from api.security import require_api_key
from api.tiers import TIER_HIGH, TIER_MEDIUM, classify

logger = logging.getLogger("finsight.routes")

router = APIRouter()

# Hard cap on batch size — protects the single-worker deployment from
# memory/CPU exhaustion via oversized payloads.
MAX_BATCH_SIZE = 500

# Drift status thresholds on |mean z-score| of recent inputs
DRIFT_WARN = 0.25
DRIFT_ALERT = 0.50


# ── Internal helpers ──────────────────────────────────────────────────────────

def _ready(request: Request):
    model = request.app.state.model
    prep = request.app.state.preprocessor
    cfg = request.app.state.cfg
    if model is None or prep is None or not cfg:
        raise HTTPException(status_code=503, detail="Model artifacts not loaded.")
    return model, prep, request.app.state.metadata, cfg


def _to_array(customers: list[CustomerInput], cfg: dict, prep: Any) -> np.ndarray:
    rows = [c.model_dump(by_alias=True) for c in customers]
    for row in rows:
        if row["pdays"] == cfg["data"]["pdays_not_contacted"]:
            row["pdays"] = cfg["data"]["pdays_fill_value"]
    return prep.transform(pd.DataFrame(rows))


def _make_prediction_dict(prob: float, threshold: float) -> dict:
    """Consistent output format for single and batch predictions."""
    subscribed = bool(prob >= threshold)
    return {
        "probability_of_subscription": round(prob, 4),
        "prediction_class": int(subscribed),
        "lead_tier": classify(prob),
        "recommendation": "Call — high-probability lead" if subscribed else "Do not call — low probability",
        "threshold_used": threshold,
    }


def _humanize_feature(name: str, cat_cols: list[str]) -> str:
    """'ohe__job_retired' -> 'job = retired'; 'num__age' -> 'age'."""
    if "__" in name:
        name = name.split("__", 1)[1]
    for col in sorted(cat_cols, key=len, reverse=True):
        prefix = col + "_"
        if name.startswith(prefix):
            return f"{col} = {name[len(prefix):]}"
    return name


def _top_drivers(request: Request, encoded: np.ndarray, k: int = 3) -> list[dict] | None:
    """Top-k SHAP feature contributions for a single encoded row.

    Returns None when the explainer is still loading or unavailable —
    prediction responses degrade gracefully.
    """
    explainer = getattr(request.app.state, "explainer", None)
    names: list[str] = request.app.state.feature_names
    if explainer is None or not names:
        return None
    try:
        sv = explainer.shap_values(encoded)
        if isinstance(sv, list):          # some shap versions: [class0, class1]
            sv = sv[-1]
        arr = np.asarray(sv)
        row = arr[0, :, -1] if arr.ndim == 3 else arr[0]

        cfg = request.app.state.cfg
        cat_cols = [c for c in cfg.get("features", {}).get("categorical", [])
                    if c != "education"]

        top = np.argsort(np.abs(row))[::-1][:k]
        return [
            {
                "feature": _humanize_feature(names[i], cat_cols),
                "impact": round(float(row[i]), 4),
                "direction": "increases" if row[i] > 0 else "decreases",
            }
            for i in top
        ]
    except Exception as e:
        logger.warning("SHAP driver computation failed: %s", e)
        return None


# ── Health & readiness ────────────────────────────────────────────────────────

@router.get("/", summary="Health check")
async def health_check(request: Request):
    """Liveness + model metadata. db_connected reflects a real (cached) ping."""
    meta = request.app.state.metadata
    return {
        "status": "active",
        "model_loaded": request.app.state.model is not None,
        "model_name": meta.get("model_name"),
        "val_pr_auc": meta.get("val_pr_auc"),
        "tuned_threshold": meta.get("tuned_threshold"),
        "trained_at": meta.get("timestamp"),
        "db_connected": await db.ping(),
    }


@router.get("/ready", summary="Component readiness")
async def readiness(request: Request):
    """Per-component readiness — RAG and SHAP load in the background after boot."""
    return {
        "model_loaded": request.app.state.model is not None,
        "explainer_ready": getattr(request.app.state, "explainer", None) is not None,
        "rag_status": getattr(request.app.state, "rag_status", "unknown"),
        "db_configured": db.configured(),
        "db_connected": await db.ping(),
    }


@router.get("/model-info", summary="Detailed model metadata")
async def model_info(request: Request):
    return request.app.state.metadata


# ── Prediction ────────────────────────────────────────────────────────────────

@router.post("/predict", summary="Single-customer subscription prediction",
             dependencies=[Depends(limit_predict)])
async def predict(customer: CustomerInput, request: Request,
                  background_tasks: BackgroundTasks):
    model, prep, meta, cfg = _ready(request)
    threshold = meta["tuned_threshold"]

    encoded = _to_array([customer], cfg, prep)
    prob = float(model.predict_proba(encoded)[0][1])

    result = _make_prediction_dict(prob, threshold)
    result["top_drivers"] = _top_drivers(request, encoded)

    # Logged after the response is sent; held by BackgroundTasks (no GC risk).
    background_tasks.add_task(
        db.log_prediction, customer.model_dump(by_alias=True), result)
    return result


@router.post("/batch-predict", summary="Batch subscription predictions",
             dependencies=[Depends(limit_batch)])
async def batch_predict(customers: list[CustomerInput], request: Request):
    if not customers:
        raise HTTPException(status_code=400, detail="Customer list must not be empty.")
    if len(customers) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"Batch too large: {len(customers)} customers "
                   f"(maximum {MAX_BATCH_SIZE} per request).",
        )

    model, prep, meta, cfg = _ready(request)
    threshold = meta["tuned_threshold"]

    encoded = _to_array(customers, cfg, prep)
    probs = model.predict_proba(encoded)[:, 1]

    return [_make_prediction_dict(float(p), threshold) for p in probs]


# ── RAG research ──────────────────────────────────────────────────────────────

@router.post("/research", summary="Financial Strategy RAG",
             dependencies=[Depends(require_api_key), Depends(limit_research)])
async def research(payload: ResearchQuery, request: Request,
                   background_tasks: BackgroundTasks):
    query = payload.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query must not be blank.")

    engine = getattr(request.app.state, "query_engine", None)
    if engine is None:
        rag_status = getattr(request.app.state, "rag_status", "unknown")
        raise HTTPException(
            status_code=503,
            detail=f"RAG engine not available ({rag_status}). Try again shortly.",
            headers={"Retry-After": "20"},
        )

    try:
        # engine.query() is synchronous (network call to OpenAI) — run it in a
        # worker thread so it cannot block the event loop for other requests.
        response = await asyncio.to_thread(engine.query, query)
    except Exception as e:
        logger.exception("RAG query failed: %s", e)
        raise HTTPException(status_code=502, detail="RAG backend error. Try again later.")

    answer = str(response)

    # Source citations — best effort, empty list on any surprise.
    sources: list[dict] = []
    try:
        seen: set[str] = set()
        for node in getattr(response, "source_nodes", []) or []:
            fname = (getattr(node, "metadata", None) or {}).get("file_name") \
                or (getattr(getattr(node, "node", None), "metadata", None) or {}).get("file_name")
            if fname and fname not in seen:
                seen.add(fname)
                score = getattr(node, "score", None)
                sources.append({
                    "file": fname,
                    "score": round(float(score), 3) if score is not None else None,
                })
            if len(sources) >= 3:
                break
    except Exception:
        sources = []

    background_tasks.add_task(db.log_research, query, answer)
    return {"query": query, "answer": answer, "sources": sources}


# ── Analytics & drift ─────────────────────────────────────────────────────────

@router.get("/analytics", summary="Aggregate usage analytics",
            dependencies=[Depends(limit_analytics)])
async def analytics():
    return await db.get_analytics()


@router.get("/drift", summary="Feature drift vs. training distribution",
            dependencies=[Depends(limit_analytics)])
async def drift(request: Request,
                n: int = Query(200, ge=20, le=1000,
                               description="How many recent predictions to analyse")):
    """Standardised mean-shift drift check for numeric features.

    Recent prediction inputs are standardised with the *training* scaler
    (mean/std stored inside preprocessor.pkl). By construction the training
    data has mean 0 per feature, so |mean(z)| of recent traffic directly
    measures distribution shift — no raw training data required.
    """
    _, prep, _, cfg = _ready(request)

    if not db.configured():
        return {"available": False, "reason": "MongoDB not configured"}

    inputs = await db.get_recent_inputs(n)
    if len(inputs) < 20:
        return {"available": False,
                "reason": f"insufficient data ({len(inputs)} samples, need >= 20)"}

    try:
        scaler = prep.named_transformers_["num"]
        num_cols: list[str] = cfg["features"]["numerical"]
        pdays_raw = cfg["data"]["pdays_not_contacted"]
        pdays_fill = cfg["data"]["pdays_fill_value"]
    except Exception:
        raise HTTPException(status_code=503, detail="Preprocessor incompatible with drift check.")

    features = []
    for i, col in enumerate(num_cols):
        alt = col.replace(".", "_")   # older logs stored underscore field names
        values = []
        for row in inputs:
            v = row.get(col, row.get(alt))
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                continue
            if col == "pdays" and v == pdays_raw:
                v = pdays_fill
            values.append(float(v))
        if len(values) < 20:
            continue
        z = (np.asarray(values) - scaler.mean_[i]) / scaler.scale_[i]
        mean_shift = float(np.abs(z.mean()))
        status = ("alert" if mean_shift >= DRIFT_ALERT
                  else "warn" if mean_shift >= DRIFT_WARN
                  else "ok")
        features.append({
            "feature": col,
            "mean_shift": round(mean_shift, 3),
            "dispersion_ratio": round(float(z.std()), 3),
            "n": len(values),
            "status": status,
        })

    return {
        "available": True,
        "n_samples": len(inputs),
        "thresholds": {"warn": DRIFT_WARN, "alert": DRIFT_ALERT},
        "tier_cutoffs": {"high": TIER_HIGH, "medium": TIER_MEDIUM},
        "features": sorted(features, key=lambda f: f["mean_shift"], reverse=True),
    }

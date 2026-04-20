"""Tests for the FastAPI application (api/main.py + api/routes.py).

   All tests are skipped when training artifacts are missing.
   Run the pipeline first:
     python -m src.data_processing
     python -m src.train
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.main import app

# ── Skip guard ────────────────────────────────────────────────────────────────

_MODELS_DIR  = Path("models")
_MODEL_READY = (
    (_MODELS_DIR / "best_model.pkl").exists()
    and (_MODELS_DIR / "preprocessor.pkl").exists()
    and (_MODELS_DIR / "best_model_metadata.json").exists()
)
_SKIP_REASON = (
    "Training artifacts not found — run:\n"
    "  python -m src.data_processing\n"
    "  python -m src.train"
)

pytestmark = pytest.mark.skipif(not _MODEL_READY, reason=_SKIP_REASON)

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def client():
    """Module-scoped TestClient — lifespan runs once for the whole test session."""
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def metadata():
    return json.loads((_MODELS_DIR / "best_model_metadata.json").read_text())


# A valid customer payload matching the raw column names the API expects
SAMPLE_CUSTOMER = {
    "age": 35, "job": "admin.", "marital": "married",
    "education": "university.degree", "default": "no",
    "housing": "yes", "loan": "no", "contact": "cellular",
    "month": "may", "day_of_week": "mon",
    "campaign": 1, "pdays": 999, "previous": 0, "poutcome": "nonexistent",
    "emp.var.rate": -1.8, "cons.price.idx": 92.893,
    "cons.conf.idx": -46.2, "euribor3m": 1.299, "nr.employed": 5099.1,
}

# ── Health / Info endpoints ───────────────────────────────────────────────────

def test_health_endpoint(client):
    """GET / returns 200 and confirms the model is loaded."""
    resp = client.get("/")
    assert resp.status_code == 200
    body = resp.json()
    assert "model_loaded" in body
    assert body["model_loaded"] is True, "Model should be loaded when artifacts exist"
    assert "model_name" in body


def test_model_info_endpoint(client, metadata):
    """GET /model-info returns the full metadata dict."""
    resp = client.get("/model-info")
    assert resp.status_code == 200
    body = resp.json()
    for key in ("model_name", "val_pr_auc", "tuned_threshold", "timestamp"):
        assert key in body, f"Missing metadata field: {key}"
    assert body["model_name"] == metadata["model_name"]

# ── /predict ─────────────────────────────────────────────────────────────────

def test_predict_valid_input(client):
    """POST /predict with valid customer data returns 200 and a probability."""
    resp = client.post("/predict", json=SAMPLE_CUSTOMER)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "probability_of_subscription" in body
    prob = body["probability_of_subscription"]
    assert 0.0 <= prob <= 1.0, f"Probability {prob} out of [0, 1]"


def test_predict_invalid_input(client):
    """POST /predict with a missing required field returns 422 Unprocessable Entity."""
    incomplete = {k: v for k, v in SAMPLE_CUSTOMER.items() if k != "age"}
    resp = client.post("/predict", json=incomplete)
    assert resp.status_code == 422


def test_predict_class_is_binary(client):
    """prediction_class in /predict response is strictly 0 or 1."""
    resp = client.post("/predict", json=SAMPLE_CUSTOMER)
    assert resp.status_code == 200
    assert resp.json()["prediction_class"] in (0, 1)


def test_predict_uses_tuned_threshold(client, metadata):
    """threshold_used in /predict response matches the saved tuned threshold."""
    resp = client.post("/predict", json=SAMPLE_CUSTOMER)
    assert resp.status_code == 200
    returned   = resp.json()["threshold_used"]
    configured = metadata["tuned_threshold"]
    assert abs(returned - configured) < 1e-6, (
        f"API threshold {returned} != metadata threshold {configured}"
    )


def test_predict_recommendation_consistent_with_class(client):
    """recommendation text matches prediction_class (call ↔ 1, do not call ↔ 0)."""
    resp = client.post("/predict", json=SAMPLE_CUSTOMER)
    body = resp.json()
    cls  = body["prediction_class"]
    rec  = body["recommendation"].lower()
    if cls == 1:
        assert "call" in rec and "do not" not in rec
    else:
        assert "do not" in rec


def test_predict_pdays_999_handled(client):
    """pdays=999 (not-previously-contacted) does not crash the API."""
    payload = {**SAMPLE_CUSTOMER, "pdays": 999}
    resp    = client.post("/predict", json=payload)
    assert resp.status_code == 200


# ── /batch-predict ────────────────────────────────────────────────────────────

def test_batch_predict_returns_correct_count(client):
    """POST /batch-predict with 3 customers returns exactly 3 predictions."""
    payload = [SAMPLE_CUSTOMER] * 3
    resp    = client.post("/batch-predict", json=payload)
    assert resp.status_code == 200
    results = resp.json()
    assert len(results) == 3


def test_batch_predict_all_probabilities_valid(client):
    """Every prediction in a batch response has a probability in [0, 1]."""
    payload = [SAMPLE_CUSTOMER] * 5
    resp    = client.post("/batch-predict", json=payload)
    for i, item in enumerate(resp.json()):
        p = item["probability_of_subscription"]
        assert 0.0 <= p <= 1.0, f"Item {i} probability {p} out of [0, 1]"


def test_batch_predict_empty_list(client):
    """POST /batch-predict with an empty list returns 400 Bad Request."""
    resp = client.post("/batch-predict", json=[])
    assert resp.status_code == 400

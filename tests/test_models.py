"""Tests for src/train.py and src/evaluate.py.

   Metric/threshold tests use synthetic data and run always.
   Artifact-dependent tests are skipped if training hasn't been run yet.
"""
from __future__ import annotations

import json
import numpy as np
import pytest
from pathlib import Path

from src.evaluate import compute_metrics
from src.train import tune_threshold, load_config

# ── Synthetic data (50 rows, used for metric/threshold tests) ─────────────────

_RNG = np.random.default_rng(42)
_N   = 50
_Y_TRUE = np.array([1] * 12 + [0] * 38)          # ~24 % positive
_Y_PROB = np.clip(_RNG.uniform(0, 1, _N), 0, 1)  # random probs

# ── Skip guard ────────────────────────────────────────────────────────────────

_MODELS_DIR = Path("models")
_MODEL_READY = (
    (_MODELS_DIR / "best_model.pkl").exists()
    and (_MODELS_DIR / "best_model_metadata.json").exists()
    and (_MODELS_DIR / "preprocessor.pkl").exists()
)
_SKIP_REASON = (
    "Training artifacts not found — run:\n"
    "  python -m src.data_processing\n"
    "  python -m src.train"
)

# ── Metric tests (no artifacts needed) ───────────────────────────────────────

def test_compute_metrics_keys():
    """compute_metrics returns all expected keys with values in [0, 1]."""
    result = compute_metrics(_Y_TRUE, _Y_PROB, threshold=0.5)
    expected_keys = {"roc_auc", "pr_auc", "f1", "precision", "recall", "accuracy"}
    assert expected_keys == set(result.keys()), f"Unexpected keys: {set(result.keys())}"
    for k, v in result.items():
        assert 0.0 <= v <= 1.0, f"Metric {k}={v} out of [0, 1]"


def test_threshold_tuning_returns_valid():
    """tune_threshold finds a threshold within the configured search grid."""
    cfg = load_config()
    ts  = cfg["models"]["threshold_search"]
    threshold = tune_threshold(_Y_TRUE, _Y_PROB, cfg)
    assert ts["grid_start"] <= threshold <= ts["grid_stop"], (
        f"Threshold {threshold} outside search grid "
        f"[{ts['grid_start']}, {ts['grid_stop']}]"
    )


def test_compute_metrics_perfect_classifier():
    """Perfect predictions yield roc_auc=1.0 and f1=1.0."""
    y = np.array([1, 1, 0, 0, 1])
    p = np.array([0.9, 0.8, 0.1, 0.2, 0.7])
    m = compute_metrics(y, p, threshold=0.5)
    assert m["roc_auc"] == 1.0
    assert m["f1"]      == 1.0


def test_compute_metrics_threshold_effect():
    """Raising the threshold increases precision at the cost of recall."""
    low  = compute_metrics(_Y_TRUE, _Y_PROB, threshold=0.2)
    high = compute_metrics(_Y_TRUE, _Y_PROB, threshold=0.8)
    assert high["precision"] >= low["precision"], "Higher threshold should not lower precision"
    assert high["recall"]    <= low["recall"],    "Higher threshold should not raise recall"


# ── Artifact-dependent tests ──────────────────────────────────────────────────

@pytest.mark.skipif(not _MODEL_READY, reason=_SKIP_REASON)
def test_best_model_artifacts_exist():
    """Required model artifacts exist on disk after training."""
    for name in ("best_model.pkl", "best_model_metadata.json",
                 "preprocessor.pkl", "feature_names.json"):
        assert (_MODELS_DIR / name).exists(), f"Missing artifact: {name}"


@pytest.mark.skipif(not _MODEL_READY, reason=_SKIP_REASON)
def test_metadata_has_required_fields():
    """best_model_metadata.json contains all required fields with valid types."""
    meta = json.loads((_MODELS_DIR / "best_model_metadata.json").read_text())
    assert "model_name"      in meta and isinstance(meta["model_name"], str)
    assert "val_pr_auc"      in meta and 0.0 <= meta["val_pr_auc"] <= 1.0
    assert "tuned_threshold" in meta and 0.0 <= meta["tuned_threshold"] <= 1.0
    assert "timestamp"       in meta and isinstance(meta["timestamp"], str)


@pytest.mark.skipif(not _MODEL_READY, reason=_SKIP_REASON)
def test_model_can_predict_single_row():
    """Loaded best_model produces a valid probability for a single preprocessed row."""
    import joblib
    model = joblib.load(_MODELS_DIR / "best_model.pkl")
    prep  = joblib.load(_MODELS_DIR / "preprocessor.pkl")
    names = json.loads((_MODELS_DIR / "feature_names.json").read_text())

    # Build a single-row array matching the preprocessor's expected output shape
    n_features = len(names)
    X_dummy = np.zeros((1, n_features))

    prob = model.predict_proba(X_dummy)[0][1]
    assert 0.0 <= prob <= 1.0, f"Predicted probability {prob} out of [0, 1]"


@pytest.mark.skipif(not _MODEL_READY, reason=_SKIP_REASON)
def test_tuned_threshold_improves_f1_over_default():
    """The tuned threshold achieves ≥ F1 of the default 0.5 threshold on test data."""
    import joblib, pandas as pd
    cfg   = load_config()
    model = joblib.load(_MODELS_DIR / "best_model.pkl")
    meta  = json.loads((_MODELS_DIR / "best_model_metadata.json").read_text())

    test_df = pd.read_parquet(Path(cfg["paths"]["processed_dir"]) / "test.parquet")
    X_test  = test_df.drop(columns=[cfg["data"]["target_col"]]).to_numpy()
    y_test  = test_df[cfg["data"]["target_col"]].to_numpy()
    y_prob  = model.predict_proba(X_test)[:, 1]

    m_tuned   = compute_metrics(y_test, y_prob, meta["tuned_threshold"])
    m_default = compute_metrics(y_test, y_prob, threshold=0.5)
    assert m_tuned["f1"] >= m_default["f1"] - 0.01, (
        f"Tuned threshold F1={m_tuned['f1']:.4f} worse than default F1={m_default['f1']:.4f}"
    )

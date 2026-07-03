from __future__ import annotations
import os
import pytest
import json
import numpy as np
import joblib
import pandas as pd
from pathlib import Path

from src.evaluate import compute_metrics
from src.train import tune_threshold, load_config

# ── Synthetic data (used for metric/threshold tests) ─────────────────
_RNG = np.random.default_rng(42)
_N   = 50
_Y_TRUE = np.array([1] * 12 + [0] * 38)
_Y_PROB = np.clip(_RNG.uniform(0, 1, _N), 0, 1)

# ── Improved Guards ──────────────────────────────────────────────────
_MODELS_DIR = Path("models")
_CONFIG = load_config()

def _is_real_artifact(path: Path) -> bool:
    """Return False for Git LFS pointer files (~130 bytes)."""
    try:
        return path.exists() and path.stat().st_size > 1024
    except Exception:
        return False

# Check for all required model artifacts (These SHOULD be in Git)
_MODEL_READY = (
    _is_real_artifact(_MODELS_DIR / "best_model.pkl")
    and (_MODELS_DIR / "best_model_metadata.json").exists()
    and _is_real_artifact(_MODELS_DIR / "preprocessor.pkl")
    and (_MODELS_DIR / "feature_names.json").exists()
)

# Check for test data (This is usually gitignored, so we skip in CI)
_TEST_DATA_PATH = Path(_CONFIG["paths"]["processed_dir"]) / "test.parquet"
_DATA_READY = _TEST_DATA_PATH.exists()

_SKIP_MODEL_REASON = "Model artifacts (pkl/json) not found in models/ folder."
_SKIP_DATA_REASON = f"Test data not found at {_TEST_DATA_PATH}. Skipping data-heavy test."

# ── Metric tests (No artifacts needed - Always Run) ──────────────────

def test_compute_metrics_keys():
    result = compute_metrics(_Y_TRUE, _Y_PROB, threshold=0.5)
    expected_keys = {"roc_auc", "pr_auc", "f1", "precision", "recall", "accuracy"}
    assert expected_keys == set(result.keys())

def test_threshold_tuning_returns_valid():
    cfg = load_config()
    ts  = cfg["models"]["threshold_search"]
    threshold = tune_threshold(_Y_TRUE, _Y_PROB, cfg)
    assert ts["grid_start"] <= threshold <= ts["grid_stop"]

# ── Artifact-dependent tests ──────────────────────────────────────────

@pytest.mark.skipif(not _MODEL_READY, reason=_SKIP_MODEL_REASON)
def test_best_model_artifacts_exist():
    for name in ("best_model.pkl", "best_model_metadata.json",
                 "preprocessor.pkl", "feature_names.json"):
        assert (_MODELS_DIR / name).exists(), f"Missing artifact: {name}"

@pytest.mark.skipif(not _MODEL_READY, reason=_SKIP_MODEL_REASON)
def test_model_can_predict_single_row():
    model = joblib.load(_MODELS_DIR / "best_model.pkl")
    names = json.loads((_MODELS_DIR / "feature_names.json").read_text())
    X_dummy = np.zeros((1, len(names)))
    prob = model.predict_proba(X_dummy)[0][1]
    assert 0.0 <= prob <= 1.0

# ── Data-dependent tests (Skipped in CI) ─────────────────────────────

@pytest.mark.skipif(not (_MODEL_READY and _DATA_READY), reason=_SKIP_DATA_REASON)
def test_tuned_threshold_improves_f1_over_default():
    model = joblib.load(_MODELS_DIR / "best_model.pkl")
    meta  = json.loads((_MODELS_DIR / "best_model_metadata.json").read_text())
    
    test_df = pd.read_parquet(_TEST_DATA_PATH)
    target = _CONFIG["data"]["target_col"]
    X_test  = test_df.drop(columns=[target]).to_numpy()
    y_test  = test_df[target].to_numpy()
    y_prob  = model.predict_proba(X_test)[:, 1]

    m_tuned   = compute_metrics(y_test, y_prob, meta["tuned_threshold"])
    m_default = compute_metrics(y_test, y_prob, threshold=0.5)
    assert m_tuned["f1"] >= m_default["f1"] - 0.01
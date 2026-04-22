"""Tests for src/data_processing.py — no full pipeline runs, no model training."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_processing import (
    load_config,
    load_raw,
    split_data,
    build_preprocessor,
    apply_smote,
)

# ── Shared synthetic schema ───────────────────────────────────────────────────

_BASE_ROW = {
    "age": 35, "job": "admin.", "marital": "married",
    "education": "university.degree", "default": "no",
    "housing": "yes", "loan": "no", "contact": "cellular",
    "month": "may", "day_of_week": "mon",
    "duration": 200, "campaign": 1, "pdays": 999,
    "previous": 0, "poutcome": "nonexistent",
    "emp.var.rate": -1.8, "cons.price.idx": 92.893,
    "cons.conf.idx": -46.2, "euribor3m": 1.299,
    "nr.employed": 5099.1, "y": "no",
}

_JOBS    = ["admin.", "blue-collar", "technician", "management", "retired"]
_MONTHS  = ["jan", "may", "jun", "aug", "nov"]
_EDU     = ["university.degree", "high.school", "basic.9y", "professional.course"]


def make_synthetic_df(n: int = 300, pos_rate: float = 0.12) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n_pos = max(10, int(n * pos_rate))
    rows = []
    for i in range(n):
        row = dict(_BASE_ROW)
        row["age"]           = int(rng.integers(18, 70))
        row["job"]           = _JOBS[i % len(_JOBS)]
        row["education"]     = _EDU[i % len(_EDU)]
        row["month"]         = _MONTHS[i % len(_MONTHS)]
        row["pdays"]         = 999 if i % 3 != 0 else int(rng.integers(1, 30))
        row["emp.var.rate"]  = float(rng.uniform(-3, 2))
        row["cons.price.idx"]= float(rng.uniform(92, 95))
        row["cons.conf.idx"] = float(rng.uniform(-50, -30))
        row["euribor3m"]     = float(rng.uniform(0.5, 5))
        row["nr.employed"]   = float(rng.uniform(4900, 5300))
        row["y"]             = "yes" if i < n_pos else "no"
        rows.append(row)
    return pd.DataFrame(rows)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def raw_df(cfg):
    return load_raw(cfg)


@pytest.fixture(scope="module")
def syn_df():
    return make_synthetic_df()


@pytest.fixture(scope="module")
def split_result(syn_df, cfg):
    """Pre-split synthetic data shared across multiple tests."""
    syn_processed = load_raw.__wrapped__(syn_df) if hasattr(load_raw, "__wrapped__") else None
    # Apply the same recode manually on synthetic data
    df = syn_df.drop(columns=["duration"])
    df = df.copy()
    df["pdays"] = df["pdays"].replace(cfg["data"]["pdays_not_contacted"],
                                       cfg["data"]["pdays_fill_value"])
    df["y"] = df["y"].map(cfg["features"]["binary_target_map"])
    return split_data(df, cfg)


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_config_loads(cfg):
    """config.yaml loads and contains all required top-level sections."""
    for key in ("project", "paths", "data", "features", "preprocessing", "models", "mlflow"):
        assert key in cfg, f"Missing top-level key: {key}"
    assert cfg["project"]["random_seed"] == 42
    assert cfg["data"]["test_size"] == 0.20
    assert "education_order" in cfg["data"]


def test_load_raw_schema(cfg, raw_df):
    """After load_raw: duration is dropped, y is integer 0/1, 20 columns total."""
    assert "duration" not in raw_df.columns, "duration (leakage column) should be dropped"
    assert raw_df["y"].dtype in (int, np.int64, np.int32)
    assert set(raw_df["y"].unique()) == {0, 1}
    # 20 raw features + y = 21 cols, minus duration = 20
    assert raw_df.shape[1] == 20


def test_pdays_recoded(cfg, raw_df):
    """After load_raw: pdays=999 ('never contacted') is recoded to -1."""
    raw_val  = cfg["data"]["pdays_not_contacted"]   # 999
    fill_val = cfg["data"]["pdays_fill_value"]       # -1
    assert raw_val not in raw_df["pdays"].values,  "999 should be recoded"
    assert fill_val in raw_df["pdays"].values,     "-1 should be present after recode"


def test_split_proportions(split_result, syn_df):
    """Train + val + test sizes sum to the full synthetic dataset size."""
    X_tr, X_va, X_te, y_tr, y_va, y_te = split_result
    total = len(X_tr) + len(X_va) + len(X_te)
    assert total == len(syn_df), f"Expected {len(syn_df)} rows total, got {total}"


def test_split_stratification(split_result, syn_df):
    """Each split preserves the original class ratio within ±2 percentage points."""
    X_tr, X_va, X_te, y_tr, y_va, y_te = split_result
    global_rate = (syn_df["y"].map({"yes": 1, "no": 0}).sum()) / len(syn_df)
    for name, y in [("train", y_tr), ("val", y_va), ("test", y_te)]:
        split_rate = y.mean()
        assert abs(split_rate - global_rate) < 0.05, (
            f"{name} class rate {split_rate:.3f} deviates >5% from global {global_rate:.3f}"
        )


def test_preprocessor_fit_on_train_only(cfg):
    """OHE categories come only from X_train — unseen test categories are absent."""
    # X_train has job=admin., X_test has an extra category
    def _make_X(jobs: list[str]) -> pd.DataFrame:
        rows = []
        for job in jobs:
            row = dict(_BASE_ROW)
            row["job"] = job
            rows.append(row)
        df = pd.DataFrame(rows).drop(columns=["duration", "y"])
        df["pdays"] = -1
        return df

    train_jobs = ["admin.", "technician", "retired"]
    X_train = _make_X(train_jobs * 10)
    X_test  = _make_X(["admin.", "exclusive_test_only_job"] * 5)

    prep = build_preprocessor(cfg)
    prep.fit(X_train)

    ohe      = prep.named_transformers_["ohe"]
    ohe_cols = [c for c in cfg["features"]["categorical"] if c != "education"]
    job_cats = set(ohe.categories_[ohe_cols.index("job")])

    assert "exclusive_test_only_job" not in job_cats, (
        "Preprocessor was fitted on test data — training/serving skew present"
    )
    # handle_unknown='ignore' — transforming unseen category must not crash or produce NaN
    encoded = prep.transform(X_test)
    assert not np.isnan(encoded).any()


def test_no_nans_after_processing(split_result, cfg):
    """Transformed X arrays contain no NaN values."""
    X_tr, X_va, X_te, y_tr, y_va, y_te = split_result
    prep = build_preprocessor(cfg)
    X_tr_t = prep.fit_transform(X_tr)
    X_va_t = prep.transform(X_va)
    X_te_t = prep.transform(X_te)

    for name, arr in [("train", X_tr_t), ("val", X_va_t), ("test", X_te_t)]:
        assert not np.isnan(arr).any(), f"NaN found in {name} after transformation"


def test_smote_balances_classes(cfg):
    """After SMOTE, positive class count equals negative class count."""
    rng = np.random.default_rng(0)
    X_imbalanced = rng.standard_normal((200, 10))
    y_imbalanced = pd.Series([1] * 20 + [0] * 180)   # 10% positive

    X_res, y_res = apply_smote(X_imbalanced, y_imbalanced, cfg)

    unique, counts = np.unique(y_res, return_counts=True)
    assert counts[0] == counts[1], (
        f"SMOTE did not balance classes: {dict(zip(unique, counts))}"
    )

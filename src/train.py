"""
Multi-model training pipeline with MLflow tracking and threshold tuning.

Trains LogisticRegression, RandomForest, XGBoost, and LightGBM on the
processed dataset. Each run:
    1. Fits on training data (with early stopping where supported)
    2. Tunes decision threshold on validation set (maximize F1)
    3. Evaluates on test set using tuned threshold
    4. Logs params, metrics, and model artifact to MLflow

The best model by val_pr_auc is saved to models/best_model.pkl
along with its metadata.

Entry point: python -m src.train
"""
from __future__ import annotations

import json
import joblib
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
from typing import Any

import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, precision_score, recall_score,
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier, early_stopping as lgbm_es, log_evaluation
from tqdm import tqdm


# ── Model Registry ────────────────────────────────────────────────────────────
# Single source of truth for display name → class + config key + excluded params.
# Adding a new model only requires a new entry here.

MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "LogisticRegression": {
        "cls": LogisticRegression,
        "config_key": "logistic_regression",
        "skip_params": set(),
    },
    "RandomForest": {
        "cls": RandomForestClassifier,
        "config_key": "random_forest",
        "skip_params": set(),
    },
    "XGBoost": {
        "cls": XGBClassifier,
        "config_key": "xgboost",
        "skip_params": set(),
    },
    "LightGBM": {
        "cls": LGBMClassifier,
        "config_key": "lightgbm",
        "skip_params": {"early_stopping_rounds"},  # passed via callbacks, not constructor
    },
}


# ── Config ────────────────────────────────────────────────────────────────────

def load_config(config_path: str | Path = "config/config.yaml") -> dict[str, Any]:
    """Load and return the YAML config as a dict."""
    with open(config_path) as f:
        return yaml.safe_load(f)


# ── Data ──────────────────────────────────────────────────────────────────────

def load_data(cfg: dict) -> tuple[np.ndarray, ...]:
    """Load processed parquet splits and return (X_tr, y_tr, X_va, y_va, X_te, y_te)."""
    proc   = Path(cfg["paths"]["processed_dir"])
    target = cfg["data"]["target_col"]

    def _split(name: str) -> tuple[np.ndarray, np.ndarray]:
        df = pd.read_parquet(proc / name)
        return df.drop(columns=[target]).to_numpy(), df[target].to_numpy()

    X_tr, y_tr = _split("train.parquet")
    X_va, y_va = _split("val.parquet")
    X_te, y_te = _split("test.parquet")
    print(f"[load]  train={X_tr.shape}  val={X_va.shape}  test={X_te.shape}")
    return X_tr, y_tr, X_va, y_va, X_te, y_te


# ── Models ────────────────────────────────────────────────────────────────────

def build_models(cfg: dict, seed: int) -> dict[str, Any]:
    """Instantiate all models from MODEL_REGISTRY using hyperparams from config."""
    def _params(config_key: str, skip: set[str]) -> dict:
        return {k: v for k, v in cfg["models"][config_key].items() if k not in skip}

    models: dict[str, Any] = {}
    for name, entry in MODEL_REGISTRY.items():
        params = _params(entry["config_key"], entry["skip_params"])
        if name == "LightGBM":
            models[name] = entry["cls"](**params, random_state=seed, verbose=-1)
        else:
            models[name] = entry["cls"](**params, random_state=seed)
    return models


# ── Metrics & Threshold ───────────────────────────────────────────────────────

def compute_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float
) -> dict[str, float]:
    """Return ROC-AUC, PR-AUC, F1, precision, and recall at the given threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "roc_auc":   round(roc_auc_score(y_true, y_prob), 4),
        "pr_auc":    round(average_precision_score(y_true, y_prob), 4),
        "f1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
    }


def tune_threshold(
    y_val: np.ndarray, y_prob_val: np.ndarray, cfg: dict
) -> float:
    """Grid-search the threshold that maximises F1 on the validation set."""
    ts = cfg["models"]["threshold_search"]
    grid = np.arange(ts["grid_start"], ts["grid_stop"], ts["grid_step"])
    best_t, best_f1 = 0.5, 0.0
    for t in grid:
        f1 = f1_score(y_val, (y_prob_val >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


# ── Training ──────────────────────────────────────────────────────────────────

def _fit(
    name: str, model: Any,
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_va: np.ndarray, y_va: np.ndarray,
    cfg: dict,
) -> None:
    """Fit one model, using early stopping for XGBoost and LightGBM."""
    if name == "XGBoost":
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    elif name == "LightGBM":
        es = cfg["models"]["lightgbm"]["early_stopping_rounds"]
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[lgbm_es(es, verbose=False), log_evaluation(period=-1)],
        )
    else:
        model.fit(X_tr, y_tr)


def train_and_log(
    name: str, model: Any,
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_va: np.ndarray, y_va: np.ndarray,
    X_te: np.ndarray, y_te: np.ndarray,
    cfg: dict,
) -> dict[str, Any]:
    """Train one model, tune threshold on val, evaluate on test, log to MLflow."""
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    with mlflow.start_run(run_name=name, tags=cfg["mlflow"]["run_tags"]):
        # ── Train ──
        _fit(name, model, X_tr, y_tr, X_va, y_va, cfg)

        # ── Val metrics + threshold tuning ──
        prob_va = model.predict_proba(X_va)[:, 1]
        threshold = tune_threshold(y_va, prob_va, cfg)
        val_metrics = compute_metrics(y_va, prob_va, threshold)

        # ── Test metrics ──
        prob_te = model.predict_proba(X_te)[:, 1]
        test_metrics = compute_metrics(y_te, prob_te, threshold)

        # ── Log to MLflow ──
        model_key = MODEL_REGISTRY[name]["config_key"]
        mlflow.log_params(cfg["models"][model_key])
        mlflow.log_metric("tuned_threshold", threshold)
        mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
        mlflow.sklearn.log_model(model, artifact_path=f"model_{name}")

    return {
        "name": name,
        "model": model,
        "threshold": threshold,
        **{f"val_{k}": v for k, v in val_metrics.items()},
        **{f"test_{k}": v for k, v in test_metrics.items()},
    }


# ── Summary & Persistence ─────────────────────────────────────────────────────

def print_summary(results: list[dict]) -> None:
    """Print a formatted comparison table of all model results."""
    cols = ["name", "val_roc_auc", "val_pr_auc", "val_f1",
            "test_roc_auc", "test_pr_auc", "test_f1", "threshold"]
    df = pd.DataFrame(results)[cols].set_index("name")
    print("\n" + "=" * 70)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 70)
    print(df.to_string())
    print("=" * 70)


def save_best(results: list[dict], cfg: dict) -> None:
    """Save the best model (by val_pr_auc) and its metadata to models/."""
    best = max(results, key=lambda r: r["val_pr_auc"])
    models_dir = Path(cfg["paths"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(best["model"], models_dir / "best_model.pkl")

    metadata = {
        "model_name":    best["name"],
        "val_pr_auc":    best["val_pr_auc"],
        "test_pr_auc":   best["test_pr_auc"],
        "tuned_threshold": best["threshold"],
        "timestamp":     datetime.now(timezone.utc).isoformat(),
    }
    (models_dir / "best_model_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"\n[best]  {best['name']}  val_pr_auc={best['val_pr_auc']:.4f}  "
          f"threshold={best['threshold']:.2f}")
    print(f"[save]  best_model.pkl  best_model_metadata.json")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(config_path: str | Path = "config/config.yaml") -> None:
    """Orchestrate the full training pipeline end-to-end."""
    cfg  = load_config(config_path)
    seed = cfg["project"]["random_seed"]
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])

    X_tr, y_tr, X_va, y_va, X_te, y_te = load_data(cfg)
    models = build_models(cfg, seed)

    results: list[dict] = []
    for name, model in tqdm(models.items(), desc="Training models"):
        print(f"\n[train] {name} ...")
        result = train_and_log(
            name, model,
            X_tr, y_tr, X_va, y_va, X_te, y_te,
            cfg,
        )
        results.append(result)
        print(f"        val_pr_auc={result['val_pr_auc']:.4f}  "
              f"test_pr_auc={result['test_pr_auc']:.4f}  "
              f"threshold={result['threshold']:.2f}")

    print_summary(results)
    save_best(results, cfg)
    print("\n[done]  training pipeline complete.")


if __name__ == "__main__":
    main()
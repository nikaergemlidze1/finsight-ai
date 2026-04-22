from __future__ import annotations

import json
import joblib
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Any

from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, precision_score,
    recall_score, accuracy_score, ConfusionMatrixDisplay, confusion_matrix,
    roc_curve, precision_recall_curve,
)


# ── Config ────────────────────────────────────────────────────────────────────

def load_config(config_path: str | Path = "config/config.yaml") -> dict[str, Any]:
    with open(config_path) as f:
        return yaml.safe_load(f)


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "roc_auc":   round(roc_auc_score(y_true, y_prob), 4),
        "pr_auc":    round(average_precision_score(y_true, y_prob), 4),
        "f1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "accuracy":  round(accuracy_score(y_true, y_pred), 4),
    }


# ── Plots ─────────────────────────────────────────────────────────────────────

def _save(fig: plt.Figure, save_path: str | Path | None) -> plt.Figure:
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
    return fig


def plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, save_path: str | Path | None = None
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred)).plot(ax=ax, colorbar=False)
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    return _save(fig, save_path)


def plot_roc_curve(
    y_true: np.ndarray, y_prob: np.ndarray, save_path: str | Path | None = None
) -> plt.Figure:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return _save(fig, save_path)


def plot_pr_curve(
    y_true: np.ndarray, y_prob: np.ndarray, save_path: str | Path | None = None
) -> plt.Figure:
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, lw=2, label=f"AP = {ap:.4f}")
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall Curve")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return _save(fig, save_path)


def plot_feature_importance(
    model: Any,
    feature_names: list[str],
    top_n: int = 20,
    save_path: str | Path | None = None,
) -> plt.Figure:
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        raise ValueError(f"Model {type(model).__name__} has no feature importance attribute.")

    idx = np.argsort(importances)[-top_n:]
    fig, ax = plt.subplots(figsize=(8, top_n * 0.35 + 1))
    ax.barh(np.array(feature_names)[idx], importances[idx])
    ax.set(title=f"Top {top_n} Feature Importances", xlabel="Importance")
    fig.tight_layout()
    return _save(fig, save_path)


def plot_threshold_analysis(
    y_true: np.ndarray, y_prob: np.ndarray, save_path: str | Path | None = None
) -> plt.Figure:
    thresholds = np.arange(0.05, 0.95, 0.01)
    metrics = {"Precision": [], "Recall": [], "F1": []}
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        metrics["Precision"].append(precision_score(y_true, y_pred, zero_division=0))
        metrics["Recall"].append(recall_score(y_true, y_pred, zero_division=0))
        metrics["F1"].append(f1_score(y_true, y_pred, zero_division=0))

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, values in metrics.items():
        ax.plot(thresholds, values, lw=2, label=label)
    ax.set(xlabel="Decision Threshold", ylabel="Score", title="Metrics vs. Threshold")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return _save(fig, save_path)


# ── CLI Entry Point ───────────────────────────────────────────────────────────

def evaluate_best_model(config_path: str | Path = "config/config.yaml") -> None:
    cfg        = load_config(config_path)
    models_dir = Path(cfg["paths"]["models_dir"])
    fig_dir    = Path(cfg["paths"]["reports_dir"])
    fig_dir.mkdir(parents=True, exist_ok=True)

    model    = joblib.load(models_dir / "best_model.pkl")
    metadata = json.loads((models_dir / "best_model_metadata.json").read_text())
    names    = json.loads((models_dir / "feature_names.json").read_text())
    threshold = metadata["tuned_threshold"]

    test_df = pd.read_parquet(Path(cfg["paths"]["processed_dir"]) / "test.parquet")
    X_test  = test_df.drop(columns=[cfg["data"]["target_col"]]).to_numpy()
    y_test  = test_df[cfg["data"]["target_col"]].to_numpy()
    y_prob  = model.predict_proba(X_test)[:, 1]
    y_pred  = (y_prob >= threshold).astype(int)

    metrics = compute_metrics(y_test, y_prob, threshold)
    print(f"\nTest metrics — {metadata['model_name']} (threshold={threshold})")
    for k, v in metrics.items():
        print(f"  {k:<12} {v:.4f}")

    plot_confusion_matrix(y_test, y_pred,       fig_dir / "confusion_matrix.png")
    plot_roc_curve(y_test, y_prob,               fig_dir / "roc_curve.png")
    plot_pr_curve(y_test, y_prob,                fig_dir / "pr_curve.png")
    plot_threshold_analysis(y_test, y_prob,      fig_dir / "threshold_analysis.png")
    plot_feature_importance(model, names,        save_path=fig_dir / "feature_importance.png")

    print(f"\n[saved] 5 plots → {fig_dir}/")


if __name__ == "__main__":
    evaluate_best_model()

"""
Feature engineering pipeline for the UCI Bank Marketing dataset.

Responsibilities:
    - Load raw CSV and drop leakage columns (duration)
    - Recode sentinel values (pdays 999 → -1)
    - Stratified train/val/test split (no encoding before split)
    - Fit ColumnTransformer on training data only
    - Apply SMOTE to training set only
    - Save processed parquets + preprocessor + feature names

Entry point: python -m src.data_processing
"""
from __future__ import annotations

import json
import joblib
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Any

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


# ── Config ────────────────────────────────────────────────────────────────────

def load_config(config_path: str | Path = "config/config.yaml") -> dict[str, Any]:
    """Load and return the YAML config as a dict."""
    with open(config_path) as f:
        return yaml.safe_load(f)


# ── Load & Clean ──────────────────────────────────────────────────────────────

def load_raw(cfg: dict) -> pd.DataFrame:
    """Read raw CSV, drop leakage columns, recode pdays, and map target to 0/1."""
    path = Path(cfg["paths"]["raw_data"])
    df = pd.read_csv(path, sep=cfg["data"]["sep"])
    print(f"[load]    raw shape: {df.shape}")

    drop_cols = [c for c in cfg["data"]["drop_cols"] if c in df.columns]
    df = df.drop(columns=drop_cols)
    print(f"[load]    dropped leakage cols: {drop_cols}")

    pdays_raw = cfg["data"]["pdays_not_contacted"]
    pdays_fill = cfg["data"]["pdays_fill_value"]
    df["pdays"] = df["pdays"].replace(pdays_raw, pdays_fill)
    print(f"[load]    pdays: recoded {pdays_raw} → {pdays_fill}")

    target_map = cfg["features"]["binary_target_map"]
    df[cfg["data"]["target_col"]] = df[cfg["data"]["target_col"]].map(target_map)
    return df


# ── Split ─────────────────────────────────────────────────────────────────────

def split_data(
    df: pd.DataFrame, cfg: dict
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
           pd.Series, pd.Series, pd.Series]:
    """Stratified train/val/test split; val is carved from the train portion."""
    target = cfg["data"]["target_col"]
    seed = cfg["project"]["random_seed"]
    stratify_col = df[target] if cfg["data"]["stratify"] else None

    X, y = df.drop(columns=[target]), df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg["data"]["test_size"],
        random_state=seed,
        stratify=stratify_col,
    )
    val_ratio = cfg["data"]["val_size"] / (1.0 - cfg["data"]["test_size"])
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=val_ratio,
        random_state=seed,
        stratify=y_train,
    )

    print(f"[split]   train={len(X_train):,}  val={len(X_val):,}  test={len(X_test):,}")
    return X_train, X_val, X_test, y_train, y_val, y_test


# ── Preprocessor ──────────────────────────────────────────────────────────────

def build_preprocessor(cfg: dict) -> ColumnTransformer:
    """Return an unfitted ColumnTransformer (StandardScaler + OrdinalEncoder + OHE)."""
    num_cols: list[str] = cfg["features"]["numerical"]
    cat_cols: list[str] = cfg["features"]["categorical"]
    edu_order: list[str] = cfg["data"]["education_order"]

    ordinal_cols = ["education"]
    ohe_cols = [c for c in cat_cols if c not in ordinal_cols]

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("ord", OrdinalEncoder(
                categories=[edu_order],
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            ), ordinal_cols),
            ("ohe", OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            ), ohe_cols),
        ],
        remainder="drop",
    )


# ── SMOTE ─────────────────────────────────────────────────────────────────────

def apply_smote(
    X: np.ndarray, y: np.ndarray | pd.Series, cfg: dict
) -> tuple[np.ndarray, np.ndarray]:
    """Oversample the minority class with SMOTE; always returns (np.ndarray, np.ndarray)."""
    smote_cfg = cfg["preprocessing"]["smote"]
    if not smote_cfg["enabled"]:
        return X, np.asarray(y)

    smote = SMOTE(
        sampling_strategy=smote_cfg["sampling_strategy"],
        k_neighbors=smote_cfg["k_neighbors"],
        random_state=cfg["project"]["random_seed"],
    )
    X_res, y_res = smote.fit_resample(X, y)
    print(f"[smote]   train before={X.shape[0]:,}  after={X_res.shape[0]:,}")
    return X_res, y_res


# ── Save ──────────────────────────────────────────────────────────────────────

def _to_parquet(X: np.ndarray, y: np.ndarray | pd.Series,
                col_names: list[str], path: Path) -> None:
    df = pd.DataFrame(X, columns=col_names)
    df["y"] = np.asarray(y)
    df.to_parquet(path, index=False)
    print(f"[save]    {path.name}: {df.shape}")


def save_artifacts(
    X_train: np.ndarray, y_train: np.ndarray | pd.Series,
    X_val: np.ndarray,   y_val: np.ndarray | pd.Series,
    X_test: np.ndarray,  y_test: np.ndarray | pd.Series,
    preprocessor: ColumnTransformer,
    feature_names: list[str],
    cfg: dict,
) -> None:
    """Persist processed parquets, preprocessor.pkl, and feature_names.json."""
    proc_dir = Path(cfg["paths"]["processed_dir"])
    models_dir = Path(cfg["paths"]["models_dir"])
    proc_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    _to_parquet(X_train, y_train, feature_names, proc_dir / "train.parquet")
    _to_parquet(X_val,   y_val,   feature_names, proc_dir / "val.parquet")
    _to_parquet(X_test,  y_test,  feature_names, proc_dir / "test.parquet")

    joblib.dump(preprocessor, models_dir / "preprocessor.pkl")
    print(f"[save]    preprocessor.pkl")

    (models_dir / "feature_names.json").write_text(json.dumps(feature_names))
    print(f"[save]    feature_names.json  ({len(feature_names)} features)")


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run_pipeline(config_path: str | Path = "config/config.yaml") -> None:
    """Orchestrate the full data engineering pipeline end-to-end."""
    cfg = load_config(config_path)

    df = load_raw(cfg)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df, cfg)

    preprocessor = build_preprocessor(cfg)
    X_train_t = preprocessor.fit_transform(X_train)   # fit ONLY on train
    X_val_t   = preprocessor.transform(X_val)
    X_test_t  = preprocessor.transform(X_test)
    print(f"[preproc] encoded shape: {X_train_t.shape[1]} features")

    feature_names: list[str] = preprocessor.get_feature_names_out().tolist()

    X_train_t, y_train_arr = apply_smote(X_train_t, y_train, cfg)

    save_artifacts(
        X_train_t, y_train_arr,
        X_val_t,   y_val,
        X_test_t,  y_test,
        preprocessor, feature_names, cfg,
    )
    print("[done]    data pipeline complete.")


if __name__ == "__main__":
    run_pipeline()

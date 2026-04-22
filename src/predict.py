from __future__ import annotations

import argparse
import json
import joblib
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any


# ── Config ────────────────────────────────────────────────────────────────────

def _load_config(config_path: str | Path = "config/config.yaml") -> dict[str, Any]:
    with open(config_path) as f:
        return yaml.safe_load(f)


# ── Artifacts ─────────────────────────────────────────────────────────────────

def load_artifacts(
    config_path: str | Path = "config/config.yaml",
) -> tuple[Any, Any, dict, dict]:
    """Return (model, preprocessor, metadata, cfg)."""
    cfg        = _load_config(config_path)
    models_dir = Path(cfg["paths"]["models_dir"])

    for name in ("best_model.pkl", "preprocessor.pkl", "best_model_metadata.json"):
        if not (models_dir / name).exists():
            raise FileNotFoundError(
                f"Missing artifact: {models_dir / name}. "
                "Run `python -m src.data_processing` then `python -m src.train`."
            )

    model    = joblib.load(models_dir / "best_model.pkl")
    prep     = joblib.load(models_dir / "preprocessor.pkl")
    metadata = json.loads((models_dir / "best_model_metadata.json").read_text())
    return model, prep, metadata, cfg


# ── Core Transform ────────────────────────────────────────────────────────────

def _to_array(
    customers: list[dict], cfg: dict, prep: Any
) -> np.ndarray:
    """Apply the same pdays recode + preprocessor as the training pipeline."""
    pdays_raw  = cfg["data"]["pdays_not_contacted"]
    pdays_fill = cfg["data"]["pdays_fill_value"]

    rows = []
    for c in customers:
        row = dict(c)
        if row.get("pdays") == pdays_raw:
            row["pdays"] = pdays_fill
        rows.append(row)

    return prep.transform(pd.DataFrame(rows))


def _format(prob: float, threshold: float) -> dict:
    subscribed = prob >= threshold
    return {
        "probability_of_subscription": round(prob, 4),
        "prediction_class": int(subscribed),
        "recommendation": "Call — high-probability lead" if subscribed else "Do not call — low probability",
        "threshold_used": threshold,
    }


# ── Public API ────────────────────────────────────────────────────────────────

def predict_one(
    customer_dict: dict,
    model: Any = None,
    prep: Any = None,
    metadata: dict | None = None,
    cfg: dict | None = None,
) -> dict:
    """Predict for a single customer dict (raw column names, dots for macro fields)."""
    if model is None:
        model, prep, metadata, cfg = load_artifacts()
    encoded = _to_array([customer_dict], cfg, prep)
    prob    = float(model.predict_proba(encoded)[0][1])
    return _format(prob, metadata["tuned_threshold"])


def predict_batch(
    customers: list[dict],
    model: Any = None,
    prep: Any = None,
    metadata: dict | None = None,
    cfg: dict | None = None,
) -> list[dict]:
    """Predict for a list of customer dicts."""
    if model is None:
        model, prep, metadata, cfg = load_artifacts()
    encoded   = _to_array(customers, cfg, prep)
    probs     = model.predict_proba(encoded)[:, 1]
    threshold = metadata["tuned_threshold"]
    return [_format(float(p), threshold) for p in probs]


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="FinSight AI — offline prediction CLI",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--json", metavar="JSON_STRING",
        help='Inline JSON, e.g. \'{"age":35,"job":"admin.","emp.var.rate":-1.8,...}\'',
    )
    group.add_argument(
        "--file", metavar="PATH",
        help="Path to a JSON file (single object or array of objects)",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()
    model, prep, metadata, cfg = load_artifacts()

    if args.json:
        payload = json.loads(args.json)
    else:
        payload = json.loads(Path(args.file).read_text())

    if isinstance(payload, list):
        results = predict_batch(payload, model, prep, metadata, cfg)
        print(json.dumps(results, indent=2))
    else:
        result = predict_one(payload, model, prep, metadata, cfg)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

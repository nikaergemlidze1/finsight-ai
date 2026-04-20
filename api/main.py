from __future__ import annotations

import json
import joblib
import yaml
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from api.routes import router


def _load_artifacts(app: FastAPI) -> None:
    models_dir = Path("models")

    with open("config/config.yaml") as f:
        app.state.cfg = yaml.safe_load(f)

    meta_path = models_dir / "best_model_metadata.json"
    model_path = models_dir / "best_model.pkl"
    prep_path  = models_dir / "preprocessor.pkl"

    missing = [p for p in (meta_path, model_path, prep_path) if not p.exists()]
    if missing:
        app.state.model        = None
        app.state.preprocessor = None
        app.state.metadata     = {}
        print(f"[startup] WARNING — artifacts not found: {[str(p) for p in missing]}")
        print("[startup] Run `python -m src.data_processing` then `python -m src.train` first.")
        return

    app.state.metadata     = json.loads(meta_path.read_text())
    app.state.model        = joblib.load(model_path)
    app.state.preprocessor = joblib.load(prep_path)
    print(f"[startup] model={app.state.metadata['model_name']}  "
          f"threshold={app.state.metadata['tuned_threshold']}  "
          f"val_pr_auc={app.state.metadata['val_pr_auc']}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_artifacts(app)
    yield  # app runs here


app = FastAPI(
    title="FinSight AI",
    description="Bank term-deposit subscription predictor.",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(router)

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

    if not all(p.exists() for p in (meta_path, model_path, prep_path)):
        app.state.model = None
        app.state.preprocessor = None
        app.state.metadata = {}
        print("[startup] WARNING: ML artifacts not found.")
        return

    app.state.metadata = json.loads(meta_path.read_text())
    app.state.model = joblib.load(model_path)
    app.state.preprocessor = joblib.load(prep_path)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Load ML Models
    _load_artifacts(app)
    
    # 2. Initialize RAG (Optional: Uncomment if your query_engine is ready)
    # from src.rag.query_engine import get_query_engine
    # app.state.query_engine = get_query_engine()
    
    yield 

app = FastAPI(
    title="FinSight AI",
    description="Financial Intelligence & Lead Scoring API",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(router)
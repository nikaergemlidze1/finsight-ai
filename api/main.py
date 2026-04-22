from __future__ import annotations
import json
import joblib
import yaml
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from api.routes import router

def _load_artifacts(app: FastAPI) -> None:
    """Loads lightweight ML models. This is fast and can stay in the main thread."""
    models_dir = Path("models")
    try:
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
        print("[startup] ML artifacts loaded successfully.")
    except Exception as e:
        print(f"[startup] Error loading ML artifacts: {e}")
        app.state.model = None

def _init_rag_engine(app: FastAPI) -> None:
    """
    Background task to initialize the RAG engine.
    This prevents the server from timing out during deployment.
    """
    try:
        from src.rag.query_engine import get_query_engine
        print("[background] Starting RAG engine initialization...")
        app.state.query_engine = get_query_engine()
        print("[background] RAG engine is now LIVE and ready for queries.")
    except (ImportError, FileNotFoundError) as e:
        app.state.query_engine = None
        print(f"[background] RAG engine skipped (not found): {e}")
    except Exception as e:
        app.state.query_engine = None
        print(f"[background] Unexpected error loading RAG: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Load ML Models (Quick)
    _load_artifacts(app)
    
    # 2. Start RAG initialization in a background thread (Slow)
    # This allows the lifespan to finish immediately, opening the port for Render.
    app.state.query_engine = None  # Default to None while loading
    rag_thread = threading.Thread(target=_init_rag_engine, args=(app,))
    rag_thread.start()
    
    yield 

app = FastAPI(
    title="FinSight AI",
    description="Financial Intelligence & Lead Scoring API",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(router)
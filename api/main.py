from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import yaml
from fastapi import FastAPI, Request

from api.routes import router

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("finsight.api")


# ── Artifact loading ──────────────────────────────────────────────────────────

def _load_artifacts(app: FastAPI) -> None:
    """Load lightweight ML artifacts. Fast — safe to run in the main thread."""
    # Safe defaults first, so a failure below can never leave app.state
    # half-initialised (which would turn 503s into raw 500s).
    app.state.cfg = {}
    app.state.model = None
    app.state.preprocessor = None
    app.state.metadata = {}
    app.state.feature_names = []

    models_dir = Path("models")
    try:
        with open("config/config.yaml") as f:
            app.state.cfg = yaml.safe_load(f)

        meta_path = models_dir / "best_model_metadata.json"
        model_path = models_dir / "best_model.pkl"
        prep_path = models_dir / "preprocessor.pkl"
        names_path = models_dir / "feature_names.json"

        if not all(p.exists() for p in (meta_path, model_path, prep_path)):
            logger.warning("ML artifacts not found — prediction endpoints will return 503.")
            return

        app.state.metadata = json.loads(meta_path.read_text())
        app.state.model = joblib.load(model_path)
        app.state.preprocessor = joblib.load(prep_path)
        if names_path.exists():
            app.state.feature_names = json.loads(names_path.read_text())
        logger.info("ML artifacts loaded successfully.")
    except Exception:
        logger.exception("Error loading ML artifacts")


def _init_heavy(app: FastAPI) -> None:
    """Background init of slow components (RAG engine, SHAP explainer).

    Runs in a daemon thread so the port opens immediately on deploy;
    endpoints report readiness via /ready until this completes.
    """
    # ── RAG engine ──
    if not os.getenv("OPENAI_API_KEY"):
        app.state.rag_status = "disabled: OPENAI_API_KEY not set"
        logger.warning("RAG engine disabled — OPENAI_API_KEY not set.")
    else:
        try:
            from src.rag.query_engine import get_query_engine
            logger.info("Starting RAG engine initialization...")
            app.state.query_engine = get_query_engine()
            app.state.rag_status = "ready"
            logger.info("RAG engine is LIVE and ready for queries.")
        except (ImportError, FileNotFoundError) as e:
            app.state.rag_status = f"failed: {e}"
            logger.warning("RAG engine skipped (not found): %s", e)
        except Exception as e:
            app.state.rag_status = "failed: unexpected error"
            logger.exception("Unexpected error loading RAG: %s", e)

    # ── SHAP explainer (best-effort; predictions work without it) ──
    if app.state.model is not None:
        try:
            import shap
            app.state.explainer = shap.TreeExplainer(app.state.model)
            logger.info("SHAP TreeExplainer ready — /predict responses include top_drivers.")
        except Exception as e:
            logger.warning("SHAP explainer unavailable (%s) — top_drivers will be null.", e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Load ML artifacts (quick)
    _load_artifacts(app)

    # 2. Defer heavy init to a background thread so the port opens immediately
    app.state.query_engine = None
    app.state.rag_status = "loading"
    app.state.explainer = None
    threading.Thread(target=_init_heavy, args=(app,), daemon=True).start()

    yield


app = FastAPI(
    title="FinSight AI",
    description="Financial Intelligence & Lead Scoring API",
    version="1.1.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def request_context(request: Request, call_next):
    """Attach a request ID and log method/path/status/duration per request."""
    request_id = request.headers.get("x-request-id", uuid.uuid4().hex[:8])
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Request-ID"] = request_id
    logger.info(
        "rid=%s %s %s -> %d (%.1f ms)",
        request_id, request.method, request.url.path,
        response.status_code, duration_ms,
    )
    return response


app.include_router(router)

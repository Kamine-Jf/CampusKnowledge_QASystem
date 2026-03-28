"""FastAPI application entrypoint."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from src.api.routes.auth import router as auth_router
from src.api.routes.chat import router as chat_router
from src.api.routes.feedback import router as feedback_router
from src.api.routes.history import router as history_router
from src.api.routes.models import router as models_router
from src.api.routes.crawler import router as crawler_router
from src.api.routes.upload import router as upload_router
from src.api.routes.warmup import router as warmup_router
from src.api.routes.admin_users import router as admin_users_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Startup / shutdown lifecycle hook."""
    # --- startup: ensure database tables exist ---
    try:
        from src.database.db_operate import (
            ensure_feedback_schema,
            ensure_model_schema,
            ensure_query_history_session_schema,
            ensure_session_title_schema,
            ensure_user_auth_schema,
        )
        ensure_user_auth_schema()
        ensure_feedback_schema()
        ensure_query_history_session_schema()
        ensure_session_title_schema()
        ensure_model_schema()
        logger.info("Database schema initialisation completed.")
    except Exception as exc:
        logger.warning("Non-fatal: DB schema init failed (%s). Some features may be unavailable.", exc)
    # --- startup: trigger background warmup to eliminate first-query cold-start ---
    try:
        from src.api.routes.warmup import trigger_warmup_background
        started = trigger_warmup_background()
        if started:
            logger.info("Background warmup triggered at server startup.")
    except Exception as exc:
        logger.warning("Non-fatal: Background warmup trigger failed (%s).", exc)
    yield
    # --- shutdown ---


app = FastAPI(
    title="Campus Knowledge QA API",
    version="1.0.0",
    description="FastAPI backend for RAG chat, session memory, and Word ingestion.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    """Simple health check endpoint."""
    return {"status": "ok"}


app.include_router(auth_router, prefix="/api", tags=["auth"])
app.include_router(chat_router, prefix="/api", tags=["chat"])
app.include_router(history_router, prefix="/api", tags=["history"])
app.include_router(upload_router, prefix="/api", tags=["upload"])
app.include_router(feedback_router, prefix="/api", tags=["feedback"])
app.include_router(models_router, prefix="/api", tags=["models"])
app.include_router(crawler_router, prefix="/api", tags=["crawler"])
app.include_router(warmup_router, prefix="/api", tags=["warmup"])
app.include_router(admin_users_router, prefix="/api", tags=["admin-users"])

STATIC_DIR = Path(__file__).resolve().parents[2] / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
def serve_index() -> HTMLResponse:
    """Serve the SPA index.html for the root path."""
    index_file = STATIC_DIR / "index.html"
    return HTMLResponse(content=index_file.read_text(encoding="utf-8"))


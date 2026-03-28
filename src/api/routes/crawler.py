"""Crawler management endpoints for admin panel."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter()

# ---- Crawler process state (module-level singleton) ----

CRAWLER_SCRIPT = Path(__file__).resolve().parents[3] / "pachong" / "zzuli_news_crawler.py"

_lock = threading.Lock()
_crawler_process: subprocess.Popen | None = None
_crawler_start_time: float | None = None
_crawler_log_lines: list[str] = []
_MAX_LOG_LINES = 200


def _read_output(proc: subprocess.Popen) -> None:
    """Background thread: read crawler stdout/stderr and store lines."""
    global _crawler_log_lines
    try:
        for raw_line in iter(proc.stdout.readline, ""):
            if raw_line == "":
                break
            line = raw_line.rstrip("\n").rstrip("\r")
            with _lock:
                _crawler_log_lines.append(line)
                if len(_crawler_log_lines) > _MAX_LOG_LINES:
                    _crawler_log_lines = _crawler_log_lines[-_MAX_LOG_LINES:]
    except Exception:
        pass


def _is_running() -> bool:
    """Check if the crawler subprocess is alive."""
    global _crawler_process, _crawler_start_time
    with _lock:
        if _crawler_process is None:
            return False
        ret = _crawler_process.poll()
        if ret is not None:
            _crawler_process = None
            _crawler_start_time = None
            return False
        return True


# ---- Response models ----

class CrawlerStatusResponse(BaseModel):
    running: bool
    pid: int | None = None
    uptime_seconds: int | None = None
    recent_logs: list[str] = []


class CrawlerActionResponse(BaseModel):
    success: bool
    message: str


# ---- Endpoints ----

@router.get("/crawler/status", response_model=CrawlerStatusResponse)
async def crawler_status() -> CrawlerStatusResponse:
    """Return current crawler process status and recent log lines."""
    running = _is_running()
    with _lock:
        pid = _crawler_process.pid if _crawler_process else None
        uptime = int(time.time() - _crawler_start_time) if _crawler_start_time and running else None
        logs = list(_crawler_log_lines[-50:])
    return CrawlerStatusResponse(running=running, pid=pid, uptime_seconds=uptime, recent_logs=logs)


@router.post("/crawler/start", response_model=CrawlerActionResponse)
async def crawler_start() -> CrawlerActionResponse:
    """Start the crawler subprocess."""
    global _crawler_process, _crawler_start_time, _crawler_log_lines

    if _is_running():
        raise HTTPException(status_code=400, detail="爬虫已在运行中")

    if not CRAWLER_SCRIPT.exists():
        raise HTTPException(status_code=500, detail=f"爬虫脚本不存在: {CRAWLER_SCRIPT}")

    try:
        with _lock:
            _crawler_log_lines = []
            _env = os.environ.copy()
            _env["PYTHONIOENCODING"] = "utf-8"
            _env["PYTHONUTF8"] = "1"
            _env["PYTHONUNBUFFERED"] = "1"
            _crawler_process = subprocess.Popen(
                [sys.executable, "-u", str(CRAWLER_SCRIPT)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=str(CRAWLER_SCRIPT.parent),
                env=_env,
            )
            _crawler_start_time = time.time()

        reader = threading.Thread(target=_read_output, args=(_crawler_process,), daemon=True)
        reader.start()

        logger.info("Crawler started, PID=%s", _crawler_process.pid)
        return CrawlerActionResponse(success=True, message=f"爬虫已启动 (PID: {_crawler_process.pid})")
    except Exception as exc:
        logger.exception("Failed to start crawler")
        raise HTTPException(status_code=500, detail=f"启动爬虫失败: {exc}") from exc


@router.post("/crawler/stop", response_model=CrawlerActionResponse)
async def crawler_stop() -> CrawlerActionResponse:
    """Stop the crawler subprocess gracefully, then force kill if needed."""
    global _crawler_process, _crawler_start_time

    if not _is_running():
        raise HTTPException(status_code=400, detail="爬虫未在运行")

    try:
        with _lock:
            proc = _crawler_process

        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)

        with _lock:
            _crawler_process = None
            _crawler_start_time = None

        logger.info("Crawler stopped.")
        return CrawlerActionResponse(success=True, message="爬虫已停止")
    except Exception as exc:
        logger.exception("Failed to stop crawler")
        raise HTTPException(status_code=500, detail=f"停止爬虫失败: {exc}") from exc

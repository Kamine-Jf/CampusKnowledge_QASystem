"""Warmup endpoint to eliminate cold-start latency after user login.

冷启动预热接口：用户登录后前端自动调用，在后台线程中预加载以下资源：
    1. RAG 模块导入（torch / sentence_transformers 等重型依赖）
    2. SentenceTransformer 向量模型加载
    3. Milvus 连接 + 集合加载
    4. MySQL 连接池首次连接
    5. OpenAI HTTP 客户端首次创建（TCP 握手）

预热全部在后台线程完成，接口立即返回，不阻塞用户操作。
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from fastapi import APIRouter

logger = logging.getLogger(__name__)
router = APIRouter()

_warmup_lock = threading.Lock()
_warmup_started = False
_warmup_done = False
_warmup_error: str | None = None
_errors_lock = threading.Lock()


def _do_warmup() -> None:
    """在后台线程中并行执行所有冷启动预加载。

    所有步骤同时启动：快速步骤（jieba/Milvus/MySQL/OpenAI）在模型加载完成前
    即可完成，确保用户第一次提问时不会遇到冷启动延迟。
    """
    global _warmup_done, _warmup_error  # pylint: disable=global-statement
    t0 = time.time()
    logger.info("🔥 [Warmup] 冷启动预热开始（并行模式）...")

    errors: list[str] = []

    def _append_error(msg: str) -> None:
        with _errors_lock:
            errors.append(msg)

    def _step_model() -> None:
        try:
            t = time.time()
            import importlib
            rag_mod = importlib.import_module("src.rag.rag_core")
            if hasattr(rag_mod, "_ensure_model"):
                rag_mod._ensure_model()
            logger.info("  ✅ 向量模型预加载完成 (%.2fs)", time.time() - t)
        except Exception as exc:
            msg = f"向量模型预加载失败: {exc}"
            logger.warning("  ⚠️ %s", msg)
            _append_error(msg)

    def _step_jieba() -> None:
        try:
            t = time.time()
            from src.rag.rag_core import _ensure_jieba
            _ensure_jieba()
            logger.info("  ✅ jieba 分词器预热完成 (%.2fs)", time.time() - t)
        except Exception as exc:
            msg = f"jieba 预热失败: {exc}"
            logger.warning("  ⚠️ %s", msg)
            _append_error(msg)

    def _step_milvus() -> None:
        try:
            t = time.time()
            from src.vector_db.milvus_operate import _get_cached_collection
            _get_cached_collection()
            logger.info("  ✅ Milvus 连接 + 集合加载完成 (%.2fs)", time.time() - t)
        except Exception as exc:
            msg = f"Milvus 预热失败: {exc}"
            logger.warning("  ⚠️ %s", msg)
            _append_error(msg)

    def _step_mysql() -> None:
        try:
            t = time.time()
            from src.database.mysql_conn import get_pooled_connection, return_pooled_connection
            conn = get_pooled_connection(use_database=True)
            return_pooled_connection(conn)
            logger.info("  ✅ MySQL 连接池预热完成 (%.2fs)", time.time() - t)
        except Exception as exc:
            msg = f"MySQL 预热失败: {exc}"
            logger.warning("  ⚠️ %s", msg)
            _append_error(msg)

    def _step_openai() -> None:
        try:
            t = time.time()
            from src.llm.qwen_operate import _get_cached_client, OPENROUTER_BASE_URL, OPENROUTER_API_KEY
            _get_cached_client(OPENROUTER_BASE_URL, OPENROUTER_API_KEY)
            logger.info("  ✅ OpenAI 客户端预热完成 (%.2fs)", time.time() - t)
        except Exception as exc:
            msg = f"OpenAI 客户端预热失败: {exc}"
            logger.warning("  ⚠️ %s", msg)
            _append_error(msg)

    with ThreadPoolExecutor(max_workers=5, thread_name_prefix="warmup-step") as executor:
        futs = [
            executor.submit(_step_model),
            executor.submit(_step_jieba),
            executor.submit(_step_milvus),
            executor.submit(_step_mysql),
            executor.submit(_step_openai),
        ]
        for fut in as_completed(futs):
            try:
                fut.result()
            except Exception:
                pass

    total = time.time() - t0
    _warmup_done = True
    _warmup_error = "; ".join(errors) if errors else None

    if errors:
        logger.warning("🔥 [Warmup] 预热完成（部分失败），总耗时 %.2fs，错误: %s", total, _warmup_error)
    else:
        logger.info("🔥 [Warmup] 冷启动预热全部完成，总耗时 %.2fs", total)


def trigger_warmup_background() -> bool:
    """在后台线程启动预热（如尚未启动）。返回 True 表示本次调用实际触发了预热。"""
    global _warmup_started  # pylint: disable=global-statement
    with _warmup_lock:
        if _warmup_started:
            return False
        _warmup_started = True
    thread = threading.Thread(target=_do_warmup, name="warmup-thread", daemon=True)
    thread.start()
    return True


@router.post("/warmup")
async def warmup() -> dict[str, object]:
    """触发冷启动预热（后台线程执行，接口立即返回）。

    前端在用户登录成功后自动调用此接口，预加载所有重型资源，
    确保用户第一次提问时无需等待冷启动。

    多次调用幂等：仅首次触发实际预热，后续调用直接返回状态。
    """
    actually_started = trigger_warmup_background()
    if not actually_started:
        return {
            "status": "already_started",
            "done": _warmup_done,
            "error": _warmup_error,
        }
    return {"status": "started", "done": False, "error": None}


@router.get("/warmup/status")
async def warmup_status() -> dict[str, object]:
    """查询预热状态。"""
    return {
        "started": _warmup_started,
        "done": _warmup_done,
        "error": _warmup_error,
    }

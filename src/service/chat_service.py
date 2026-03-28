"""Chat service that calls RAG pipeline and stores session memory."""

from __future__ import annotations

import asyncio
import importlib
from datetime import datetime
from uuid import uuid4

from src.service.memory_service import normalize_session_id, save_session_turn
from src.database.db_operate import get_model_by_id, ensure_model_schema, list_query_history_by_session


def _get_rag_query_fn():
    """Lazy-import the heavy RAG module so the server can start without ML libs."""
    mod = importlib.import_module("src.rag.rag_core")
    return mod.stage4_rag_query


def _get_rag_stream_fn():
    """Lazy-import the streaming RAG function."""
    mod = importlib.import_module("src.rag.rag_core")
    return mod.stage4_rag_query_stream


def _build_session_id(session_id: str | None) -> str:
    """Reuse incoming session id or create a new one."""
    if session_id and session_id.strip():
        return normalize_session_id(session_id)
    return str(uuid4())


def _resolve_model_config(model_id: int | None) -> dict | None:
    """Look up model config from DB if model_id is provided."""
    if model_id is None:
        return None
    try:
        ensure_model_schema()
        row = get_model_by_id(model_id)
        if row:
            return {
                "model_name": row["model_id"],
                "api_base": row["api_base"],
                "api_key": row.get("api_key", ""),
            }
    except Exception:
        pass
    return None


# ===================== 多轮对话历史加载 =====================
# 最多携带最近 MAX_HISTORY_TURNS 轮对话作为上下文，避免 token 溢出
MAX_HISTORY_TURNS = 5


def _load_conversation_history(session_id: str | None) -> list[dict[str, str]]:
    """从数据库加载当前会话的历史对话，转换为 LLM messages 格式。

    返回:
        list[dict]: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
        空列表表示无历史（新会话或加载失败）
    """
    if not session_id or not session_id.strip():
        return []
    try:
        records = list_query_history_by_session(session_id, limit=MAX_HISTORY_TURNS)
    except Exception:
        return []
    messages: list[dict[str, str]] = []
    for record in records:
        q = (record.get("query_content") or "").strip()
        a = (record.get("answer_content") or "").strip()
        if q:
            messages.append({"role": "user", "content": q})
        if a:
            messages.append({"role": "assistant", "content": a})
    return messages


async def chat_once(query: str, session_id: str | None = None, model_id: int | None = None, user_id: int | None = None) -> dict[str, object]:
    """Execute one chat turn and persist history."""
    normalized_query = (query or "").strip()
    if not normalized_query:
        raise ValueError("query cannot be empty")

    normalized_session = _build_session_id(session_id)
    model_config = _resolve_model_config(model_id)
    history = _load_conversation_history(normalized_session)
    rag_fn = _get_rag_query_fn()
    answer = await asyncio.to_thread(rag_fn, normalized_query, model_config, history)
    save_session_turn(normalized_session, normalized_query, answer, user_id=user_id)

    return {
        "session_id": normalized_session,
        "query": normalized_query,
        "answer": answer,
        "created_at": datetime.now(),
    }


async def chat_once_stream(query: str, session_id: str | None = None, model_id: int | None = None, user_id: int | None = None):
    """Execute one streaming chat turn. Yields (event, data) tuples for SSE."""
    import json

    normalized_query = (query or "").strip()
    if not normalized_query:
        raise ValueError("query cannot be empty")

    normalized_session = _build_session_id(session_id)
    model_config = _resolve_model_config(model_id)
    history = _load_conversation_history(normalized_session)

    # 先发送 session_id 让前端知道会话ID
    yield ("session", json.dumps({"session_id": normalized_session}))

    # 在线程中执行同步的流式生成器
    rag_stream_fn = _get_rag_stream_fn()
    full_answer_parts = []

    # 使用 asyncio.Queue 替代 threading.Queue，避免轮询延迟
    import threading

    aq: asyncio.Queue = asyncio.Queue()
    sentinel = object()
    loop = asyncio.get_running_loop()

    def _run_stream():
        try:
            for chunk in rag_stream_fn(normalized_query, model_config, history):
                loop.call_soon_threadsafe(aq.put_nowait, chunk)
        except Exception as e:
            loop.call_soon_threadsafe(aq.put_nowait, e)
        finally:
            loop.call_soon_threadsafe(aq.put_nowait, sentinel)

    thread = threading.Thread(target=_run_stream, daemon=True)
    thread.start()

    while True:
        item = await aq.get()

        if item is sentinel:
            break
        if isinstance(item, Exception):
            yield ("error", json.dumps({"detail": str(item)}))
            break

        full_answer_parts.append(item)
        yield ("chunk", item)

    # 流式完成后保存完整回答
    full_answer = "".join(full_answer_parts).strip()
    if full_answer:
        try:
            save_session_turn(normalized_session, normalized_query, full_answer, user_id=user_id)
        except Exception as e:
            print(f"⚠️ 保存会话记录失败：{e}")

    yield ("done", json.dumps({"session_id": normalized_session, "query": normalized_query}))

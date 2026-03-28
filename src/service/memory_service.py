"""Session memory service backed by MySQL history table."""

from __future__ import annotations

from typing import Any

from src.database.db_operate import (
    add_query_history_with_session,
    delete_session,
    ensure_query_history_session_schema,
    ensure_session_title_schema,
    list_query_history_by_session,
    list_sessions,
    set_session_custom_title,
)


MAX_SESSION_ID_LENGTH = 64


def normalize_session_id(session_id: str | None) -> str:
    """Normalize and validate session id."""
    normalized = (session_id or "").strip()
    if not normalized:
        raise ValueError("session_id cannot be empty")
    if len(normalized) > MAX_SESSION_ID_LENGTH:
        raise ValueError(f"session_id length must be <= {MAX_SESSION_ID_LENGTH}")
    return normalized


def save_session_turn(session_id: str, query: str, answer: str, user_id: int | None = None) -> bool:
    """Persist one query/answer turn for a session."""
    normalized_session = normalize_session_id(session_id)
    normalized_query = (query or "").strip()
    normalized_answer = (answer or "").strip()

    if not normalized_query:
        raise ValueError("query cannot be empty")
    if not normalized_answer:
        raise ValueError("answer cannot be empty")

    ensure_query_history_session_schema()
    return add_query_history_with_session(
        session_id=normalized_session,
        query_content=normalized_query,
        answer_content=normalized_answer,
        user_id=user_id,
    )


def list_session_history(session_id: str, limit: int = 100) -> dict[str, Any]:
    """Fetch history items by session id."""
    normalized_session = normalize_session_id(session_id)
    if limit < 1:
        raise ValueError("limit must be >= 1")

    ensure_query_history_session_schema()
    records = list_query_history_by_session(session_id=normalized_session, limit=limit)
    return {
        "session_id": normalized_session,
        "total": len(records),
        "items": records,
    }


def list_all_sessions(limit: int = 50, user_id: int | None = None) -> dict[str, Any]:
    """Return all unique sessions for sidebar display, filtered by user_id."""
    ensure_query_history_session_schema()
    rows = list_sessions(limit=limit, user_id=user_id)
    items = []
    for row in rows:
        raw = str(row.get("first_query") or "新对话")
        title = raw[:50] if len(raw) > 50 else raw
        items.append({
            "session_id": row["session_id"],
            "title": title,
            "last_active": row["last_active"],
        })
    return {"total": len(items), "items": items}


def rename_session(session_id: str, title: str) -> bool:
    """Persist a user-defined custom title for a session."""
    normalized_session = normalize_session_id(session_id)
    normalized_title = title.strip()[:200]
    if not normalized_title:
        raise ValueError("title cannot be empty")
    ensure_session_title_schema()
    return set_session_custom_title(normalized_session, normalized_title)


def delete_user_session(session_id: str, user_id: int | None = None) -> bool:
    """Delete all records for a session, optionally restricted to a user."""
    normalized_session = normalize_session_id(session_id)
    ensure_query_history_session_schema()
    return delete_session(session_id=normalized_session, user_id=user_id)

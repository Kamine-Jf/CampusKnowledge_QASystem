"""Feedback service for user feedback submission and admin retrieval."""

from __future__ import annotations

from typing import Any

from src.database.db_operate import (
    add_feedback,
    del_feedback,
    ensure_feedback_schema,
    list_feedback,
)


def submit_feedback(
    feedback_text: str,
    user_id: int | None = None,
    user_name: str = "",
    query_content: str = "",
    answer_content: str = "",
) -> dict[str, Any]:
    """Submit user feedback."""
    normalized_text = (feedback_text or "").strip()
    if not normalized_text:
        raise ValueError("feedback_text cannot be empty")

    ensure_feedback_schema()
    success = add_feedback(
        feedback_text=normalized_text,
        user_id=user_id,
        user_name=(user_name or "").strip(),
        query_content=(query_content or "").strip(),
        answer_content=(answer_content or "").strip(),
    )
    if not success:
        raise RuntimeError("failed to save feedback")

    return {"success": True, "message": "feedback submitted"}


def get_all_feedback() -> dict[str, Any]:
    """Get all feedback records for admin."""
    ensure_feedback_schema()
    records = list_feedback()
    return {"total": len(records), "items": records}


def delete_feedback(record_id: int) -> dict[str, Any]:
    """Delete a feedback record by id."""
    success = del_feedback(record_id)
    if not success:
        raise RuntimeError(f"feedback record {record_id} not found or delete failed")
    return {"success": True, "message": "feedback deleted"}

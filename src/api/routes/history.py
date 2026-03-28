"""History endpoint for session conversation retrieval."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from src.api.schemas.chat import HistoryResponse, SessionListResponse
from src.service.memory_service import list_session_history, list_all_sessions, delete_user_session, rename_session


router = APIRouter()


@router.get("/sessions", response_model=SessionListResponse)
async def get_sessions(
    limit: int = Query(default=50, ge=1, le=200),
    user_id: int | None = Query(default=None, description="Filter sessions by user id"),
) -> SessionListResponse:
    """Return list of all chat sessions for sidebar, filtered by user_id."""
    try:
        payload = list_all_sessions(limit=limit, user_id=user_id)
    except Exception as service_error:  # pylint: disable=broad-except
        raise HTTPException(status_code=500, detail=f"sessions service error: {service_error}") from service_error

    return SessionListResponse(**payload)


@router.get("/history/{session_id}", response_model=HistoryResponse)
async def get_history(session_id: str, limit: int = Query(default=100, ge=1, le=500)) -> HistoryResponse:
    """Return stored dialog records for one session id."""
    try:
        payload = list_session_history(session_id=session_id, limit=limit)
    except ValueError as value_error:
        raise HTTPException(status_code=400, detail=str(value_error)) from value_error
    except Exception as service_error:  # pylint: disable=broad-except
        raise HTTPException(status_code=500, detail=f"history service error: {service_error}") from service_error

    return HistoryResponse(**payload)


@router.patch("/sessions/{session_id}/title")
async def rename_session_endpoint(session_id: str, body: dict) -> dict:
    """Set a custom title for a session."""
    title = (body.get("title") or "").strip()
    if not title:
        raise HTTPException(status_code=400, detail="title cannot be empty")
    try:
        success = rename_session(session_id=session_id, title=title)
    except ValueError as value_error:
        raise HTTPException(status_code=400, detail=str(value_error)) from value_error
    except Exception as service_error:  # pylint: disable=broad-except
        raise HTTPException(status_code=500, detail=f"rename session error: {service_error}") from service_error
    if not success:
        raise HTTPException(status_code=500, detail="failed to rename session")
    return {"success": True, "session_id": session_id, "title": title}


@router.delete("/sessions/{session_id}")
async def delete_session_endpoint(
    session_id: str,
    user_id: int | None = Query(default=None, description="Restrict deletion to this user"),
) -> dict:
    """Delete all records for a session."""
    try:
        success = delete_user_session(session_id=session_id, user_id=user_id)
    except ValueError as value_error:
        raise HTTPException(status_code=400, detail=str(value_error)) from value_error
    except Exception as service_error:  # pylint: disable=broad-except
        raise HTTPException(status_code=500, detail=f"delete session error: {service_error}") from service_error

    if not success:
        raise HTTPException(status_code=404, detail="session not found or already deleted")
    return {"success": True, "message": "session deleted"}

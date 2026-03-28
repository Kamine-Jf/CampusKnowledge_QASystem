"""Admin user management endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from src.database.db_operate import (
    delete_session,
    delete_user_by_id,
    delete_user_history_by_user_id,
    list_all_users,
    list_query_history_by_session,
    list_user_sessions,
)

router = APIRouter()


@router.get("/admin/users")
async def get_all_users(limit: int = Query(default=200, ge=1, le=500)) -> dict:
    """Return all registered users."""
    try:
        users = list_all_users(limit=limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    return {"total": len(users), "items": users}


@router.get("/admin/users/{user_id}/sessions")
async def get_user_sessions(user_id: int, limit: int = Query(default=50, ge=1, le=200)) -> dict:
    """Return all chat sessions for a specific user."""
    try:
        sessions = list_user_sessions(user_id=user_id, limit=limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    items = []
    for s in sessions:
        first_query = str(s.get("first_query") or "新对话")
        items.append({
            "session_id": s["session_id"],
            "title": first_query[:30] if len(first_query) > 30 else first_query,
            "message_count": s.get("message_count", 0),
            "last_active": s.get("last_active"),
        })
    return {"user_id": user_id, "total": len(items), "items": items}


@router.get("/admin/users/{user_id}/sessions/{session_id}/messages")
async def get_session_messages(user_id: int, session_id: str, limit: int = Query(default=100, ge=1, le=500)) -> dict:
    """Return all messages in a specific session."""
    try:
        records = list_query_history_by_session(session_id=session_id, limit=limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    return {"session_id": session_id, "total": len(records), "items": records}


@router.delete("/admin/users/{user_id}/sessions/{session_id}")
async def delete_user_session_admin(user_id: int, session_id: str) -> dict:
    """Delete a specific session for a user."""
    try:
        success = delete_session(session_id=session_id, user_id=user_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    if not success:
        raise HTTPException(status_code=404, detail="会话不存在或已删除")
    return {"success": True, "message": "会话已删除"}


@router.delete("/admin/users/{user_id}/history")
async def delete_all_user_history(user_id: int) -> dict:
    """Delete all chat history for a user."""
    try:
        delete_user_history_by_user_id(user_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    return {"success": True, "message": "该用户全部对话历史已删除"}


@router.delete("/admin/users/{user_id}")
async def delete_user(user_id: int) -> dict:
    """Delete (deactivate) a user account and all their history."""
    try:
        delete_user_history_by_user_id(user_id)
        success = delete_user_by_id(user_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    if not success:
        raise HTTPException(status_code=404, detail="用户不存在或已注销")
    return {"success": True, "message": "用户已注销，相关数据已清除"}

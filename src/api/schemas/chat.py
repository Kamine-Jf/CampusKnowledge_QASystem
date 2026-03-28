"""Schemas for chat and history APIs."""



from __future__ import annotations



from datetime import datetime



from pydantic import BaseModel, Field





class ChatRequest(BaseModel):

    """Request payload for chat API."""



    query: str = Field(..., min_length=1, max_length=2000, description="User question")

    session_id: str | None = Field(default=None, max_length=64, description="Optional session id")

    model_id: int | None = Field(default=None, description="Optional LLM model ID from admin config")

    user_id: int | None = Field(default=None, description="Logged-in user ID")





class ChatResponse(BaseModel):

    """Response payload for chat API."""



    session_id: str

    query: str

    answer: str

    created_at: datetime





class HistoryItem(BaseModel):

    """Single history message record."""



    id: int

    session_id: str

    query_content: str

    answer_content: str

    create_time: datetime





class HistoryResponse(BaseModel):

    """History list for a given session id."""



    session_id: str

    total: int

    items: list[HistoryItem]



class SessionItem(BaseModel):

    """Single session summary for sidebar list."""

    session_id: str

    title: str

    last_active: datetime



class SessionListResponse(BaseModel):

    """Session list response for sidebar."""

    total: int

    items: list[SessionItem]


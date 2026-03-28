"""Schemas for feedback APIs."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class FeedbackRequest(BaseModel):
    """Request payload for submitting feedback."""

    feedback_text: str = Field(..., min_length=1, max_length=2000, description="Feedback content")
    user_id: int | None = Field(default=None, description="User id")
    user_name: str = Field(default="", max_length=50, description="User name")
    query_content: str = Field(default="", max_length=2000, description="Original question")
    answer_content: str = Field(default="", max_length=5000, description="AI answer")


class FeedbackResponse(BaseModel):
    """Response payload for submitting feedback."""

    success: bool
    message: str


class FeedbackItem(BaseModel):
    """Single feedback record."""

    id: int
    user_id: int | None
    user_name: str
    query_content: str
    answer_content: str
    feedback_text: str
    created_at: datetime


class FeedbackListResponse(BaseModel):
    """Response payload for listing feedback."""

    total: int
    items: list[FeedbackItem]

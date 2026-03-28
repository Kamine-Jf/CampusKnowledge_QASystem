"""Feedback endpoints for user submission and admin retrieval."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from src.api.schemas.feedback import (
    FeedbackListResponse,
    FeedbackRequest,
    FeedbackResponse,
)
from src.service.feedback_service import delete_feedback, get_all_feedback, submit_feedback


router = APIRouter()


@router.post("/feedback", response_model=FeedbackResponse)
async def post_feedback(request: FeedbackRequest) -> FeedbackResponse:
    """Submit user feedback on an AI answer."""
    try:
        result = submit_feedback(
            feedback_text=request.feedback_text,
            user_id=request.user_id,
            user_name=request.user_name,
            query_content=request.query_content,
            answer_content=request.answer_content,
        )
    except ValueError as value_error:
        raise HTTPException(status_code=400, detail=str(value_error)) from value_error
    except Exception as service_error:  # pylint: disable=broad-except
        raise HTTPException(status_code=500, detail=f"feedback service error: {service_error}") from service_error

    return FeedbackResponse(**result)


@router.get("/admin/feedback", response_model=FeedbackListResponse)
async def admin_feedback_list() -> FeedbackListResponse:
    """Admin endpoint to list all user feedback."""
    try:
        result = get_all_feedback()
    except Exception as service_error:  # pylint: disable=broad-except
        raise HTTPException(status_code=500, detail=f"feedback list error: {service_error}") from service_error

    return FeedbackListResponse(**result)


@router.delete("/admin/feedback/{feedback_id}", response_model=FeedbackResponse)
async def admin_delete_feedback(feedback_id: int) -> FeedbackResponse:
    """Admin endpoint to delete a feedback record by id."""
    try:
        result = delete_feedback(feedback_id)
    except RuntimeError as not_found_error:
        raise HTTPException(status_code=404, detail=str(not_found_error)) from not_found_error
    except Exception as service_error:  # pylint: disable=broad-except
        raise HTTPException(status_code=500, detail=f"feedback delete error: {service_error}") from service_error

    return FeedbackResponse(**result)

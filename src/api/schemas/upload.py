"""Schemas for document upload API."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class UploadResponse(BaseModel):
    """Response payload for docx upload endpoint."""

    file_name: str
    chunks: int
    inserted_count: int
    save_path: str


class UploadedDocumentItem(BaseModel):
    """Single uploaded document item for admin table."""

    id: int
    file_name: str
    file_type: str
    save_path: str
    uploaded_at: datetime


class UploadedDocumentListResponse(BaseModel):
    """Uploaded document list payload."""

    total: int
    items: list[UploadedDocumentItem]

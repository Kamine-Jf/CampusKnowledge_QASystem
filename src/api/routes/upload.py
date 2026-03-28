"""Upload endpoint for excel/pdf/word parsing and ingestion."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from src.api.schemas.upload import UploadResponse, UploadedDocumentListResponse
from src.service.doc_service import ingest_file_upload, list_uploaded_documents, delete_uploaded_document

_TEMPLATE_PATH = Path(__file__).resolve().parents[3] / "data" / "structured_data" / "模板.xlsx"


router = APIRouter()


@router.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)) -> UploadResponse:
    """Upload excel/pdf/word file, parse and ingest into MySQL or Milvus."""
    try:
        result = await ingest_file_upload(file)
    except ValueError as value_error:
        raise HTTPException(status_code=400, detail=str(value_error)) from value_error
    except Exception as service_error:  # pylint: disable=broad-except
        raise HTTPException(status_code=500, detail=f"upload service error: {service_error}") from service_error

    return UploadResponse(**result)


@router.get("/upload/list", response_model=UploadedDocumentListResponse)
async def upload_list() -> UploadedDocumentListResponse:
    """List uploaded documents for admin table."""
    try:
        items = list_uploaded_documents()
    except Exception as service_error:  # pylint: disable=broad-except
        raise HTTPException(status_code=500, detail=f"upload list service error: {service_error}") from service_error

    return UploadedDocumentListResponse(total=len(items), items=items)


@router.get("/upload/template")
async def download_template() -> FileResponse:
    """Download the Excel structured-data template file."""
    if not _TEMPLATE_PATH.exists():
        raise HTTPException(status_code=404, detail="模板文件不存在")
    return FileResponse(
        path=str(_TEMPLATE_PATH),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename="模板.xlsx",
    )


@router.delete("/upload/{doc_id}")
async def delete_doc(doc_id: int) -> dict:
    """Delete an uploaded document record by id."""
    try:
        success = delete_uploaded_document(doc_id)
    except Exception as service_error:  # pylint: disable=broad-except
        raise HTTPException(status_code=500, detail=f"delete error: {service_error}") from service_error
    if not success:
        raise HTTPException(status_code=404, detail="document not found")
    return {"success": True, "message": "deleted"}

"""Models endpoints for CRUD management of LLM models."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.database.db_operate import (
    ensure_model_schema,
    add_model,
    list_models as db_list_models,
    del_model,
    update_model,
    get_model_by_id,
)


router = APIRouter()


class ModelAddRequest(BaseModel):
    """Request to add a new LLM model."""
    model_name: str = Field(..., min_length=1, max_length=100, description="模型显示名称")
    model_id: str = Field(..., min_length=1, max_length=200, description="API模型标识符")
    api_base: str = Field(default="https://openrouter.ai/api/v1", max_length=500, description="API基础地址")
    api_key: str = Field(default="", max_length=500, description="API密钥")
    is_default: int = Field(default=0, ge=0, le=1, description="是否为默认模型")


class ModelUpdateRequest(BaseModel):
    """Request to update a model."""
    model_name: str = Field(..., min_length=1, max_length=100)
    model_id: str = Field(..., min_length=1, max_length=200)
    api_base: str = Field(default="https://openrouter.ai/api/v1", max_length=500)
    api_key: str = Field(default="", max_length=500)
    is_default: int = Field(default=0, ge=0, le=1)
    enabled: int = Field(default=1, ge=0, le=1)


@router.get("/models")
async def list_models_endpoint() -> dict:
    """Return available model options from database."""
    ensure_model_schema()
    rows = db_list_models(enabled_only=False)
    models = []
    default_id = None
    for row in rows:
        m = {
            "id": row["id"],
            "model_id": row["model_id"],
            "name": row["model_name"],
            "api_base": row["api_base"],
            "api_key_set": bool(row.get("api_key")),
            "is_default": bool(row.get("is_default")),
            "enabled": bool(row.get("enabled", 1)),
            "created_at": row.get("created_at"),
        }
        models.append(m)
        if row.get("is_default"):
            default_id = row["id"]
    return {"models": models, "default": default_id}


@router.post("/models")
async def add_model_endpoint(req: ModelAddRequest) -> dict:
    """Add a new LLM model configuration."""
    ensure_model_schema()
    success = add_model(
        model_name=req.model_name,
        model_id=req.model_id,
        api_base=req.api_base,
        api_key=req.api_key,
        is_default=req.is_default,
    )
    if not success:
        raise HTTPException(status_code=500, detail="添加模型失败")
    return {"success": True, "message": "模型添加成功"}


@router.get("/models/{record_id}")
async def get_model_endpoint(record_id: int) -> dict:
    """Return full details of a single model configuration (for editing)."""
    ensure_model_schema()
    model = get_model_by_id(record_id)
    if not model:
        raise HTTPException(status_code=404, detail="模型不存在")
    return {
        "id": model["id"],
        "model_name": model["model_name"],
        "model_id": model["model_id"],
        "api_base": model["api_base"],
        "api_key": model.get("api_key", ""),
        "is_default": int(bool(model.get("is_default"))),
        "enabled": int(bool(model.get("enabled", 1))),
        "created_at": model.get("created_at"),
    }


@router.put("/models/{record_id}")
async def update_model_endpoint(record_id: int, req: ModelUpdateRequest) -> dict:
    """Update an existing model configuration."""
    ensure_model_schema()
    existing = get_model_by_id(record_id)
    if not existing:
        raise HTTPException(status_code=404, detail="模型不存在")
    success = update_model(
        record_id=record_id,
        model_name=req.model_name,
        model_id=req.model_id,
        api_base=req.api_base,
        api_key=req.api_key,
        is_default=req.is_default,
        enabled=req.enabled,
    )
    if not success:
        raise HTTPException(status_code=500, detail="更新模型失败")
    return {"success": True, "message": "模型更新成功"}


@router.delete("/models/{record_id}")
async def delete_model_endpoint(record_id: int) -> dict:
    """Delete a model configuration."""
    ensure_model_schema()
    success = del_model(record_id)
    if not success:
        raise HTTPException(status_code=404, detail="模型不存在")
    return {"success": True, "message": "模型已删除"}


@router.post("/models/{record_id}/default")
async def set_default_model(record_id: int) -> dict:
    """Set a model as the default."""
    ensure_model_schema()
    existing = get_model_by_id(record_id)
    if not existing:
        raise HTTPException(status_code=404, detail="模型不存在")
    # Clear all defaults first, then set the chosen one
    all_models = db_list_models(enabled_only=False)
    for m in all_models:
        if m["is_default"] and m["id"] != record_id:
            update_model(
                record_id=m["id"],
                model_name=m["model_name"],
                model_id=m["model_id"],
                api_base=m["api_base"],
                api_key=m.get("api_key", ""),
                is_default=0,
                enabled=m.get("enabled", 1),
            )
    update_model(
        record_id=record_id,
        model_name=existing["model_name"],
        model_id=existing["model_id"],
        api_base=existing["api_base"],
        api_key=existing.get("api_key", ""),
        is_default=1,
        enabled=existing.get("enabled", 1),
    )
    return {"success": True, "message": "已设为默认模型"}

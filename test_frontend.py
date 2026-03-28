"""Minimal test server to verify frontend renders correctly without DB deps."""
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

app = FastAPI()

STATIC_DIR = Path(__file__).resolve().parent / "static"

# In-memory mock data
_mock_models = [
    {"id": 1, "model_id": "arcee-ai/trinity-large-preview:free", "model_name": "Trinity-Large", "name": "Trinity-Large", "api_base": "https://openrouter.ai/api/v1", "api_key_set": True, "is_default": True, "enabled": True, "created_at": "2025-03-25T10:00:00"},
]
_next_model_id = 2

# API mock endpoints
@app.post("/api/auth/login")
async def mock_login():
    return {"success": True, "message": "login success", "user": {"id": 1, "name": "张三", "student_id": "2021001234", "phone": "13800138000", "created_at": "2025-01-01T00:00:00"}}

@app.post("/api/auth/register")
async def mock_register():
    return {"success": True, "message": "register success", "user": {"id": 2, "name": "Test", "student_id": "2021005678", "phone": "13900139000", "created_at": "2025-01-01T00:00:00"}}

@app.post("/api/auth/admin/login")
async def mock_admin_login():
    return {"success": True, "message": "admin login success", "role": "admin"}

@app.post("/api/chat")
async def mock_chat(request: Request):
    body = await request.json()
    mid = body.get("model_id")
    model_name = "默认模型"
    for m in _mock_models:
        if m["id"] == mid:
            model_name = m["name"]
    return {"session_id": "test-session-1", "query": body.get("query", ""), "answer": f"[{model_name}] 这是一个测试回答，系统运行正常。", "created_at": "2025-01-01T00:00:00"}

@app.get("/api/sessions")
async def mock_sessions():
    return {"total": 0, "items": []}

@app.get("/api/history/{session_id}")
async def mock_history(session_id: str):
    return {"session_id": session_id, "total": 0, "items": []}

@app.get("/api/upload/list")
async def mock_upload_list():
    return {"total": 1, "items": [{"id": 1, "file_name": "test.pdf", "file_type": "PDF", "save_path": "/data/test.pdf", "uploaded_at": "2025-03-25T10:00:00"}]}

@app.delete("/api/upload/{doc_id}")
async def mock_delete_doc(doc_id: int):
    return {"success": True, "message": "deleted"}

@app.get("/api/admin/feedback")
async def mock_feedback():
    return {"total": 0, "items": []}

@app.get("/api/models")
async def mock_models():
    default_id = None
    for m in _mock_models:
        if m.get("is_default"):
            default_id = m["id"]
    return {"models": _mock_models, "default": default_id}

@app.post("/api/models")
async def mock_add_model(request: Request):
    global _next_model_id
    body = await request.json()
    m = {
        "id": _next_model_id,
        "model_id": body.get("model_id", ""),
        "model_name": body.get("model_name", ""),
        "name": body.get("model_name", ""),
        "api_base": body.get("api_base", ""),
        "api_key_set": bool(body.get("api_key")),
        "is_default": bool(body.get("is_default")),
        "enabled": True,
        "created_at": "2025-03-25T12:00:00",
    }
    _mock_models.append(m)
    _next_model_id += 1
    return {"success": True, "message": "模型添加成功"}

@app.get("/api/models/{record_id}")
async def mock_get_model(record_id: int):
    for m in _mock_models:
        if m["id"] == record_id:
            return {
                "id": m["id"],
                "model_name": m.get("model_name", m.get("name", "")),
                "model_id": m["model_id"],
                "api_base": m.get("api_base", ""),
                "api_key": "",
                "is_default": int(bool(m.get("is_default"))),
                "enabled": int(bool(m.get("enabled", True))),
            }
    from fastapi import HTTPException
    raise HTTPException(status_code=404, detail="模型不存在")


@app.delete("/api/models/{record_id}")
async def mock_delete_model(record_id: int):
    global _mock_models
    _mock_models = [m for m in _mock_models if m["id"] != record_id]
    return {"success": True, "message": "模型已删除"}

@app.post("/api/models/{record_id}/default")
async def mock_set_default(record_id: int):
    for m in _mock_models:
        m["is_default"] = (m["id"] == record_id)
    return {"success": True, "message": "已设为默认模型"}

@app.put("/api/models/{record_id}")
async def mock_update_model(record_id: int, request: Request):
    body = await request.json()
    for m in _mock_models:
        if m["id"] == record_id:
            m.update({"model_name": body.get("model_name", m["model_name"]), "name": body.get("model_name", m["name"]), "model_id": body.get("model_id", m["model_id"]), "api_base": body.get("api_base", m["api_base"])})
    return {"success": True, "message": "模型更新成功"}

# Static files & SPA
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/", response_class=HTMLResponse)
def serve_index():
    return HTMLResponse(content=(STATIC_DIR / "index.html").read_text(encoding="utf-8"))

if __name__ == "__main__":
    uvicorn.run("test_frontend:app", host="0.0.0.0", port=8080, reload=False)

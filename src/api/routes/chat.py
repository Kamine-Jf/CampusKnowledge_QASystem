"""Chat endpoint that orchestrates RAG and session memory."""



from __future__ import annotations



from fastapi import APIRouter, HTTPException, Request

from fastapi.responses import StreamingResponse



from src.api.schemas.chat import ChatRequest, ChatResponse

from src.service.chat_service import chat_once, chat_once_stream





router = APIRouter()





@router.post("/chat", response_model=ChatResponse)

async def chat(request: ChatRequest) -> ChatResponse:

    """Run one chat turn and persist into session history."""

    query = (request.query or "").strip()

    if not query:

        raise HTTPException(status_code=400, detail="query cannot be empty")



    try:

        result = await chat_once(query=query, session_id=request.session_id, model_id=request.model_id, user_id=request.user_id)

    except ValueError as value_error:

        raise HTTPException(status_code=400, detail=str(value_error)) from value_error

    except Exception as service_error:  # pylint: disable=broad-except

        raise HTTPException(status_code=500, detail=f"chat service error: {service_error}") from service_error



    return ChatResponse(**result)


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest, raw_request: Request):
    """Stream one chat turn via Server-Sent Events."""
    query = (request.query or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="query cannot be empty")

    def _sse_frame(event_type: str, data: str) -> str:
        """Build a spec-compliant SSE frame.

        SSE requires that newlines inside data be sent as separate
        ``data:`` lines within the same event.  A blank line terminates
        the frame.
        """
        lines = data.split("\n")
        data_part = "\n".join(f"data: {line}" for line in lines)
        return f"event: {event_type}\n{data_part}\n\n"

    async def event_generator():
        try:
            async for event_type, data in chat_once_stream(
                query=query,
                session_id=request.session_id,
                model_id=request.model_id,
                user_id=request.user_id,
            ):
                # 检查客户端是否断开连接（用户点击停止生成）
                if await raw_request.is_disconnected():
                    print("🛑 客户端已断开，停止流式生成")
                    break
                yield _sse_frame(event_type, data)
        except ValueError as e:
            yield _sse_frame("error", str(e))
        except Exception as e:
            yield _sse_frame("error", str(e))

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

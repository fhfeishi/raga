# backend/api/routes.py
from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse
from services.chatflow_services import ChatFlowService

router = APIRouter(prefix="/api")

# 全局共享服务实例
chat_service = ChatFlowService()

# ------------------- SSE 流式对话接口 -------------------
@router.get("/chat/stream")
async def stream_chat(question: str = Query(..., description="用户问题")):
    async def event_stream():
        try:
            for token in chat_service.sse_stream(question):
                yield f"data: {token}\n\n"
        except Exception as e:
            yield f"data: [ERROR] {str(e)}\n\n"
        yield "data: [END]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
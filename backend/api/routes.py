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
    def event_stream():
        # 让浏览器/代理不要缓存、尽快开始显示
        yield "retry: 150000\n\n"  # 可选：告诉客户端断线重连间隔
        try:
            for token in chat_service.token_stream(question):
                yield f"data: {token}\n\n"
        except Exception as e:
            yield f"event: error\ndata: {str(e)}\n\n"
        finally:
            yield "data: [END]\n\n"

    return StreamingResponse(
        event_stream(), 
        media_type="text/event-stream",
        headers={
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        # 若日后挂到 Nginx 等反代后面，建议加这一条禁用缓冲：
        "X-Accel-Buffering": "no",
        },)
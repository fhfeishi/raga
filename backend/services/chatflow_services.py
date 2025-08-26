# services/chatflow_services.py
"""
ChatFlow 服务层：封装 RAGAgent，提供适配 FastAPI 的接口
支持：普通流式、SSE 流式
"""

from agents.rag_agent import RAGAgent
from typing import Iterator
from pydantic import BaseModel

class ChatFlowService:
    def __init__(self, agent: RAGAgent = None):
        self.agent = agent or RAGAgent()

    def token_stream(self, question: str) -> Iterator[str]:
        """
        生成 token 流（用于 CLI 或内部调用）
        """
        for token in self.agent.stream_tokens(question):
            yield str(token)
        yield "\n"

    def sse_stream(self, question: str) -> Iterator[str]:
        """
        生成 SSE 流（用于 FastAPI 返回 text/event-stream）
        格式: data: xxx\n\n
        结束: data: [END]\n\n
        """
        for token in self.agent.stream_tokens(question):
            yield f"data: {token}\n\n"
    
    
# query-message 的包装器    
class ChatBody(BaseModel):
    messages        : list              # 
    use_context     : bool = False      # todo.
    context_filter  : bool = False      # todo.
    include_sources : bool = True       # 溯源
    stream          : bool = True       # 
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a rapper. Always answer with a rap.",
                        },
                        {
                            "role": "user",
                            "content": "How do you fry an egg?",
                        },
                    ],
                    "stream": False,
                    "use_context": True,
                    "include_sources": True,
                    "context_filter": {
                        "docs_ids": ["c202d5e6-7b69-4869-81cc-dd574ee8ee11"]
                    },
                }
            ]
        }
    }
    

    
        
# 可在 chatflow_service.py 底部加：
if __name__ == "__main__":
    service = ChatFlowService()
    print(">>> RAG Agent 已启动，输入问题（exit 退出）")
    import sys
    for line in sys.stdin:
        q = line.strip()
        if q.lower() in {"exit", "quit"}:
            break
        print("助手: ", end="")
        for tok in service.token_stream(q):
            print(tok, end="", flush=True)
        print()
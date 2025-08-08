# agents/rag_agent.py
"""
RAG Agent 模块：负责检索与生成的核心逻辑
支持 HuggingFace / Ollama 模型
支持流式输出，可过滤 <think> 等中间推理标记
"""


from langchain_core.prompts import ChatPromptTemplate
import time 
from typing import Iterator
from models.chat_model import chat_model_cratefn
from models.embedding_model import enbedding_model_cratefn
from data_db.vector_db import vector_store_cratefn
from agents.prompts import SYSTEM_PROMPT

import logging 
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')


class RAGAgent:
    def __init__(self, mode='auto'):

        # 初始化嵌入模型
        self.embeddings = enbedding_model_cratefn()

        # 初始化向量数据库
        self.vectordb = vector_store_cratefn()

        # 初始化 LLM
        self.llm = chat_model_cratefn()

        # 初始化提示词
        self.prompt = self._create_prompt(prompt=SYSTEM_PROMPT)

 
    def _create_prompt(self, prompt) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages([
            ("system", prompt),
            ("human", "{question}")
        ])

    def retrieve_context(self, question: str, k: int = 2) -> str:
        """检索相关文档片段"""
        docs = self.vectordb.similarity_search(question, k=k)
        return "\n\n".join(d.page_content for d in docs)

    def _stream_response(self, question: str, k: int = 2, hide_think: bool = True) -> Iterator[str]:
        """
        流式生成回答，可选择是否过滤 <think> 等中间推理标记
        流式生成回答，兼容 Linux/Ubuntu 缓冲问题
        """
        t0 = time.perf_counter()
        context = self.retrieve_context(question, k=k)
        logger.info(f"🔍 检索耗时: {(time.perf_counter() - t0)*1000:.0f} ms")
        logger.debug(f"检索到的上下文: {context}")

        # 构造消息
        messages = self.prompt.format_prompt(question=question, context=context).to_messages()

        first_token_time = None
        if hide_think:
            START_TAG = "</think>"
            MAX_TAG = 10
            ring = ""
            in_answer = False

            for chunk in self.llm.stream(messages):
                content = chunk.content or ""
                if not content:
                    continue

                ring += content
                if len(ring) > MAX_TAG:
                    ring = ring[-MAX_TAG:]

                if not in_answer:
                    idx = ring.lower().find(START_TAG)
                    if idx != -1:
                        post = ring[idx + len(START_TAG):]
                        if post:
                            yield post
                            if first_token_time is None:
                                first_token_time = time.perf_counter()
                        in_answer = True
                        ring = ""
                    continue

                # 已进入回答，正常输出
                if content:
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                    yield content
        else:
            for chunk in self.llm.stream(messages):
                content = chunk.content or ""
                if content:
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                    yield content

        total_time = time.perf_counter() - t0
        first_token_latency = (first_token_time - t0) * 1000 if first_token_time else 0
        logger.info(f"⏱️ 首 token 延迟: {first_token_latency:.0f} ms | 总耗时: {total_time*1000:.0f} ms")

    def stream_tokens(self, question: str, k: int = 2, hide_think: bool = True) -> Iterator[str]:
        """对外暴露的流式 token 生成接口"""
        yield from self._stream_response(question, k=k, hide_think=hide_think)
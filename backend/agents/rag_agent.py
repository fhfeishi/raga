# agents/rag_agent.py
"""
RAG Agent 模块：负责检索与生成的核心逻辑
支持 HuggingFace / Ollama 模型
支持流式输出，可过滤 <think> 等中间推理标记
"""

from langchain_core.prompts import ChatPromptTemplate
import time 
from typing import Iterator, List 
from models.chat_model import chat_model_cratefn
from models.embedding_model import enbedding_model_cratefn
from data_db.vector_db import vector_store_cratefn
from models.rerank_model import rerank_model_createfn
from agents.prompts import SYSTEM_PROMPT

import logging 
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')


class RAGAgent:
    def __init__(
        self, 
        mode: str='auto',
        use_rerank: bool = True,
        pre_k: int=20, # 粗排召回数量（向量检索）
        top_k: int=4   # 精排保留数量（交叉编码器）
        ):
        
        """
        :param use_rerank: 是否启用交叉编码器 Re-rank
        :param pre_k: 向量粗排召回数量
        :param top_k: 交叉编码器精排后保留数量（进入 prompt 的文档数）
        """

        # 初始化嵌入模型
        self.embeddings = enbedding_model_cratefn()

        # 初始化向量数据库
        self.vectordb = vector_store_cratefn()

        # 初始化 LLM
        self.llm = chat_model_cratefn()

        # 初始化提示词
        self.prompt = self._create_prompt(prompt=SYSTEM_PROMPT)

        self.use_rerank = use_rerank 
        self.pre_k = pre_k
        self.top_k = top_k 
        
        self.reranker = None 
        if self.use_rerank:
            try:
                self.reranker = rerank_model_createfn()
                logger.info("🔧 Reranker 已启用")
            except Exception as e:
                logger.warning(f"⚠️ Reranker 加载失败，降级为纯向量检索: {e}")
                self.reranker = None
        
        
 
    def _create_prompt(self, prompt) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages([
            ("system", prompt),
            ("human", "{question}")
        ])

    # ----- 检错（支持粗排+精排）
    def retrieve_docs(self, question: str) -> List:
        t0 = time.perf_counter()
        # 1) 向量粗排
        rough_docs = self.vectordb.similarity_search(question, k=self.pre_k)
        t_vec = (time.perf_counter() - t0) * 1000
        logger.info(f"🔎 向量检索: {len(rough_docs)} 条，用时 {t_vec:.0f} ms")

        # 2) 交叉编码器精排
        if self.reranker is not None and rough_docs:
            try: 
                t1 = time.perf_counter()
                ranked = self.reranker.rerank(
                    question, 
                    rough_docs, 
                    top_k=self.top_k, 
                    with_score=True,
                    batch_size=8)  #  1 8 16
                t_rerank = (time.perf_counter() - t1) * 1000
                logger.info(f"🏁 Re-rank 完成: 取 Top-{self.top_k}，用时 {t_rerank:.0f} ms")
                # 只返回文档，必要时你也可以把分数带回去做调试
                return [item["doc"] for item in ranked]
            except Exception as e:
                logger.warning(f"⚠️ Re-rank 失败，降级纯向量检索: {e}")
            
        else:
            # 无 reranker 时，直接取前 top_k
            return rough_docs[: self.top_k]

    def retrieve_context(self, question: str) -> str:
        """拼接文档内容为单一上下文字符串"""
        docs = self.retrieve_docs(question)
        return "\n\n".join(getattr(d, "page_content", str(d)) for d in docs)

    # ------ 流式 生成
    def _stream_response(self, question: str, hide_think: bool = True) -> Iterator[str]:
        """
        流式生成回答，可选择是否过滤 <think> 等中间推理标记
        流式生成回答，兼容 Linux/Ubuntu 缓冲问题
        """
        t0 = time.perf_counter()
        context = self.retrieve_context(question)
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

    def stream_tokens(self, question: str, hide_think: bool = True) -> Iterator[str]:
        """对外暴露的流式 token 生成接口"""
        yield from self._stream_response(question, hide_think=hide_think)

# settings.py

"""  
全局配置入口   —— 单文件强类型配置
目标：
- 本地hf-models优先
- 可切换向量库（默认 Chroma，可扩展 Qdrant/FAISS）
- 与 LlamaIndex 解耦（通过 build_* 工厂延迟导入，适配 0.9/0.10+/2025 版本差异）
- 统一入口：settings.configure_llama_index() 一步生效
"""

import os 
from pathlib import Path
from typing import Any, Dict, Literal, Optional
from pydantic import BaseModel, Field, field_validator, ConfigDict, model_validator 



# 基础路径 & 日志
# ============ 基础路径 & 日志 ============
class AppPaths(BaseModel):
    model_config = ConfigDict(validate_assignment=True)  # 赋值也走校验
    project_root: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[1])
    data_dir: Path = Field(default_factory=lambda: Path("./data"))
    cache_dir: Path = Field(default_factory=lambda: Path("./.cache"))
    persist_dir: Path = Field(default_factory=lambda: Path("./storage"))
    logs_dir: Path = Field(default_factory=lambda: Path("./logs"))

    @field_validator("data_dir", "cache_dir", "persist_dir", "logs_dir", mode="before")
    def _ensure_dir(cls, v):
        p = Path(v)
        p.mkdir(parents=True, exist_ok=True)
        return p
    
    # @model_validator(mode="after")
    # def _mkdirs(self):
    #     for p in (self.data_dir, self.cache_dir, self.persist_dir, self.logs_dir):
    #         Path(p).mkdir(parents=True, exist_ok=True)
    #     return self
    

class LoggingCfg(BaseModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    json: bool = False


# ============ 模型配置（本地优先）  ============
# ok-0903
class ChatModelCfg(BaseModel):
    load_method   : Literal["hf_cache"]    = "hf_cache"  # 目前仅支持本地缓存加载
    model_name    : str                    = "Qwen/Qwen3-1.7B"
    context_window: int                    = 8192
    max_tokens    : int                    = 400
    system_prompt : Optional[str]          = None  # 可选的系统提示词
    cache_folder  : str                    = "E:/local_models/huggingface/cache/hub"  # g
    device        : Literal["cpu", "cuda"] = 'cpu'         # 先就cpu吧 
    temperature   : float                  = 0.2
    top_k         : int                    = 50      
    top_p         : float                  = 0.7     
    do_sample     : bool                   = True

# ok -0903
class embeddingCfg(BaseModel):
    # embedding-model cfg
    load_method   : Literal["hf_cache"]    = "hf_cache"     # 目前仅支持本地缓存加载
    model_name    : str                    = "Qwen/Qwen3-Embedding-0.6B"
    max_length    : Optional[int]          = 1024  
    normalize     : bool                   = True
    embed_batch_size: int                  = 16             
    cache_folder  : str                    = "E:/local_models/huggingface/cache/hub"           
    trust_remote_code: bool                = True           
    local_files_only : bool                = False          
    device        : Literal["cpu", "cuda"] = 'cpu'         # 先就cpu吧 
    num_workers   : int                    = 4               # DataLoader 线程数，0表示不使用多线程
    # embedding_ingest cfg
    ingest_mode     : Literal["batch", "simple"] = "batch"  
    count_workers   : int                        = 4




class RerankerModelCfg(BaseModel):
    load_method   : Literal["hf_cache"]    = "hf_cache"  # 目前仅支持本地缓存加载
    model_name    : str                    = "【示意】bge-reranker-v2-m3"
    cache_folder  : str                    = "E:/local_models/huggingface/cache/hub" 
    device        : str                    = "cuda"
    with_score    : bool                   = True
    top_n         : int                    = 20


# ============ 文本分块 / 检索 ============
class SplitterCfg(BaseModel):
    type: Literal["sentence", "recursive", "token"] = "sentence"
    chunk_size: int = 1024
    chunk_overlap: int = 100
    # 针对 sentence splitter：窗口增强、标题融合等可在后续实现中使用
    add_title: bool = True


class RetrievalCfg(BaseModel):
    top_k: int = 5
    mmr: bool = True
    mmr_diversity: float = 0.3
    similarity: Literal["cosine", "dot", "euclidean"] = "cosine"


# ============ 向量库 ============
class ChromaCfg(BaseModel):
    collection: str = "private_gpt"
    persist_dir: Optional[Path] = None  # 默认使用 AppPaths.persist_dir / "chroma"


class QdrantCfg(BaseModel):
    host: str = "127.0.0.1"
    port: int = 6333
    collection: str = "private_gpt"
    prefer_grpc: bool = False


class VectorStoreCfg(BaseModel):
    backend: Literal["chroma", "qdrant", "faiss"] = "chroma"
    chroma: ChromaCfg = Field(default_factory=ChromaCfg)
    qdrant: QdrantCfg = Field(default_factory=QdrantCfg)



# ============ 汇总 App 配置 ============
class AppSettings(BaseModel):
    paths: AppPaths = Field(default_factory=AppPaths)
    logging: LoggingCfg = Field(default_factory=LoggingCfg)
    chat_model: ChatModelCfg = Field(default_factory=ChatModelCfg)
    embedding: embeddingCfg = Field(default_factory=embeddingCfg)
    reranker_model: RerankerModelCfg = Field(default_factory=RerankerModelCfg)
    splitter: SplitterCfg = Field(default_factory=SplitterCfg)
    retrieval: RetrievalCfg = Field(default_factory=RetrievalCfg)
    vector_store: VectorStoreCfg = Field(default_factory=VectorStoreCfg)

    # ---------- 工厂：延迟导入，适配不同版本的 llama-index ----------
    def _import(self, path_variants: list[str]):
        """尝试多条导入路径，适配 llama-index 不同版本目录结构。"""
        last_err = None
        for p in path_variants:
            try:
                module_path, name = p.rsplit(".", 1)
                mod = __import__(module_path, fromlist=[name])
                return getattr(mod, name)
            except Exception as e:  # pragma: no cover
                last_err = e
        raise last_err

    # ---- 构建 embedding ----
    def build_embedding(self):
        embed_v  = self.embedding_model.load_method
        # 默认就是fh默认缓存的路径
        if  embed_v == "hf_cache": 
            EmbCls = self._import([
                "llama_index.embeddings.huggingface.HuggingFaceEmbedding",
            ])
            return EmbCls(
                model_name=self.embedding_model.model_name,
                max_length=self.embedding_model.max_length,
                normalize=self.embedding_model.normalize,
                embed_batch_size=self.embedding_model.embed_batch_size,
                cache_folder=self.embedding_model.cache_folder,
                trust_remote_code=self.embedding_model.trust_remote_code,
                model_kwargs={"local_files_only": self.embedding_model.local_files_only},
                device=self.embedding_model.device,
                num_workers=self.embedding_model.num_workers,
            )

        else:
            raise NotImplementedError(f"Unknown embedding provider: {embed_v}")

    # ---- 构建 reranker（可选） ----
    def build_reranker(self):
        if self.reranker_model.provider == "none":
            return None
        if self.reranker_model.provider == "bge_reranker":
            # FlagEmbeddingReranker / BGERerankers 的路径在不同版本可能不同
            try:
                RerankCls = self._import([
                    "llama_index.postprocessor.flag_embedding_reranker.FlagEmbeddingReranker",
                    "llama_index.core.postprocessor.flag_embedding_reranker.FlagEmbeddingReranker",
                ])
                return RerankCls(
                    model=self.reranker_model.model,
                    top_n=self.reranker_model.top_n,
                    device=self.reranker_model.device,
                    **self.reranker_model.extra,
                )
            except Exception:
                # 兼容其他 cross-encoder 风格
                CrossEncCls = self._import([
                    "llama_index.postprocessor.cross_encoder_rerankers.CrossEncoderReranker",
                    "llama_index.core.postprocessor.cross_encoder_rerankers.CrossEncoderReranker",
                ])
                return CrossEncCls(
                    model=self.reranker_model.model,
                    top_n=self.reranker_model.top_n,
                    device=self.reranker_model.device,
                    **self.reranker_model.extra,
                )
        elif self.reranker_model.provider == "cross_encoder":
            CrossEncCls = self._import([
                "llama_index.postprocessor.cross_encoder_rerankers.CrossEncoderReranker",
                "llama_index.core.postprocessor.cross_encoder_rerankers.CrossEncoderReranker",
            ])
            return CrossEncCls(
                model=self.reranker_model.model,
                top_n=self.reranker_model.top_n,
                device=self.reranker_model.device,
                **self.reranker_model.extra,
            )
        else:
            raise NotImplementedError(f"Unknown reranker provider: {self.reranker_model.provider}")

    # ---- 构建 chat LLM ----
    def build_chat_llm(self):
        chat_v = self.chat_model.load_method
        if chat_v == "hf_cache":
            LLMCls = self._import([
                "llama_index.llms.huggingface.HuggingFaceLLM",
            ])
            return LLMCls(
                model_name=self.chat_model.model_name,
                tokenizer_name=self.chat_model.model_name,
                context_window=self.chat_model.context_window,
                max_new_tokens=self.chat_model.max_tokens,
                system_prompt=self.chat_model.system_prompt,
                generate_kwargs={"temperature": self.chat_model.temperature, 
                                 "top_k": self.chat_model.top_k, 
                                 "top_p": self.chat_model.top_p, 
                                 "do_sample": True},
                device_map=self.chat_model.device,
            )
        else:
            raise NotImplementedError(f"Unknown chat provider: {chat_v}")

    # ---- 构建 splitter ----
    def build_text_splitter(self):
        if self.splitter.type == "sentence":
            Splitter = self._import([
                "llama_index.core.node_parser.SentenceSplitter",
                "llama_index.node_parser.SentenceSplitter",
                "llama_index.text_splitter.SentenceSplitter",
            ])
            return Splitter(
                chunk_size=self.splitter.chunk_size,
                chunk_overlap=self.splitter.chunk_overlap,
                include_prev_next_rel=True,
            )
        elif self.splitter.type == "recursive":
            Splitter = self._import([
                "llama_index.core.node_parser.TextSplitter",
                "llama_index.text_splitter.TextSplitter",
            ])
            return Splitter(
                chunk_size=self.splitter.chunk_size,
                chunk_overlap=self.splitter.chunk_overlap,
            )
        elif self.splitter.type == "token":
            Splitter = self._import([
                "llama_index.core.node_parser.TokenTextSplitter",
                "llama_index.text_splitter.TokenTextSplitter",
            ])
            return Splitter(
                chunk_size=self.splitter.chunk_size,
                chunk_overlap=self.splitter.chunk_overlap,
            )
        else:
            raise NotImplementedError(self.splitter.type)

    # ---- 构建向量库适配（返回 LlamaIndex VectorStore） ----
    def build_vector_store(self):
        if self.vector_store.backend == "chroma":
            VectorStore = self._import([
                "llama_index.vector_stores.chroma.ChromaVectorStore",
                "llama_index.core.vector_stores.chroma.ChromaVectorStore",
            ])
            import chromadb  # 直接依赖 Chroma 客户端
            persist = self.vector_store.chroma.persist_dir or (self.paths.persist_dir / "chroma")
            persist.mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(path=str(persist))
            coll = client.get_or_create_collection(self.vector_store.chroma.collection)
            return VectorStore(chroma_collection=coll)
        elif self.vector_store.backend == "qdrant":
            VectorStore = self._import([
                "llama_index.vector_stores.qdrant.QdrantVectorStore",
                "llama_index.core.vector_stores.qdrant.QdrantVectorStore",
            ])
            from qdrant_client import QdrantClient
            qc = QdrantClient(host=self.vector_store.qdrant.host, port=self.vector_store.qdrant.port, prefer_grpc=self.vector_store.qdrant.prefer_grpc)
            return VectorStore(client=qc, collection_name=self.vector_store.qdrant.collection)
        elif self.vector_store.backend == "faiss":
            VectorStore = self._import([
                "llama_index.vector_stores.faiss.FaissVectorStore",
                "llama_index.core.vector_stores.faiss.FaissVectorStore",
            ])
            return VectorStore()
        else:
            raise NotImplementedError(self.vector_store.backend)

    # ---- 一键应用到 LlamaIndex 全局 Settings ----
    def configure_llama_index(self):
        LI_Settings = self._import([
            "llama_index.core.Settings",
            "llama_index.Settings",
        ])
        embed = self.build_embedding()
        llm = self.build_chat_llm()
        splitter = self.build_text_splitter()
        reranker = self.build_reranker()
        LI_Settings.embed_model = embed
        LI_Settings.llm = llm
        LI_Settings.text_splitter = splitter
        # reranker 不是全局 Setting，通常在 Retriever 或 QueryEngine 中作为后处理；保留工厂引用即可
        return {
            "embed_model": embed,
            "llm": llm,
            "text_splitter": splitter,
            "reranker": reranker,
        }


# 单例配置对象，可在任意模块导入
settings_ = AppSettings()


if __name__ == "__main__":
    objs = settings_.configure_llama_index()
    vs = settings_.build_vector_store()
    print("Settings OK →", {
        "llm": type(objs["llm"]).__name__,
        "embed": type(objs["embed_model"]).__name__,
        "splitter": type(objs["text_splitter"]).__name__,
        "vstore": type(vs).__name__,
    })
















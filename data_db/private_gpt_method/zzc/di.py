# private_gpt_method/di.py
from __future__ import annotations
from injector import Injector, Module, provider, singleton
from pathlib import Path

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore

import chromadb

from .settings import settings
from .settings_paths import ensure_dirs
from typing import Any

class EmbeddingComponent:
    def __init__(self) -> None:
        cfg = settings().hf
        model_kwargs: dict[str, Any] = {}
        if cfg.cache_dir:
            model_kwargs["cache_folder"] = cfg.cache_dir
        # llama-index 的 HF 封装内部会调用 transformers / sentence-transformers
        self.embedding_model = HuggingFaceEmbedding(
            model_name=cfg.model_name_or_path,
            device=cfg.device,
            trust_remote_code=cfg.trust_remote_code,
            # ↓ 传给 sentence-transformers/transformers 的参数
            model_kwargs={
                "local_files_only": cfg.local_files_only,
                "cache_dir": cfg.cache_dir,
            },
            # encode_kwargs 可根据模型需要扩展
        )

class VectorStoreComponent:
    def __init__(self) -> None:
        cfg = settings().vs
        assert cfg.backend == "chroma", "当前最小实现仅支持 chroma"
        ensure_dirs()
        self._client = chromadb.PersistentClient(path=cfg.chroma_dir)
        self._collection = self._client.get_or_create_collection(cfg.collection)
        self.vector_store = ChromaVectorStore(client=self._client, collection_name=cfg.collection)

class NodeStoreComponent:
    """文档/索引持久化；与 StorageContext 一起 persist."""
    def __init__(self) -> None:
        persist = settings().storage.persist_dir
        ensure_dirs()
        self.persist_dir = Path(persist)
        # LlamaIndex 简单存储（可持久化）
        self.doc_store = SimpleDocumentStore()
        self.index_store = SimpleIndexStore()

class PgmModule(Module):
    @singleton
    @provider
    def provide_embedding_component(self) -> EmbeddingComponent:
        return EmbeddingComponent()

    @singleton
    @provider
    def provide_vector_store_component(self) -> VectorStoreComponent:
        return VectorStoreComponent()

    @singleton
    @provider
    def provide_node_store_component(self) -> NodeStoreComponent:
        return NodeStoreComponent()

def build_injector() -> Injector:
    return Injector([PgmModule()])

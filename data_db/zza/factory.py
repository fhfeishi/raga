# factory.py
from private_gpt.components.embedding.embedding_component import EmbeddingComponent
from private_gpt.components.llm.llm_component import LLMComponent
from private_gpt.components.node_store.node_store_component import NodeStoreComponent
from private_gpt.components.vector_store.vector_store_component import VectorStoreComponent

from .service import IngestService


def build_default_ingest_service() -> IngestService:
    """不使用 injector，直接按默认构造组件（组件内部读 settings）。"""
    return IngestService(
        llm_component=LLMComponent(),
        vector_store_component=VectorStoreComponent(),
        embedding_component=EmbeddingComponent(),
        node_store_component=NodeStoreComponent(),
    )

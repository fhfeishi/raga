# private_gpt_method/di.py
from __future__ import annotations
from injector import Injector, Module, provider, singleton

# 直接复用原仓库的组件（最稳的方式）
from private_gpt.components.embedding.embedding_component import EmbeddingComponent
from private_gpt.components.llm.llm_component import LLMComponent
from private_gpt.components.node_store.node_store_component import NodeStoreComponent
from private_gpt.components.vector_store.vector_store_component import VectorStoreComponent

from .ingest import IngestService


class IngestModule(Module):
    @singleton
    @provider
    def provide_llm_component(self) -> LLMComponent:
        return LLMComponent()

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

    @singleton
    @provider
    def provide_ingest_service(
        self,
        llm: LLMComponent,
        vs: VectorStoreComponent,
        emb: EmbeddingComponent,
        ns: NodeStoreComponent,
    ) -> IngestService:
        return IngestService(
            llm_component=llm, 
            vector_store_component=vs, 
            embedding_component=emb, 
            node_store_component=ns
        )


def build_injector() -> Injector:
    return Injector([IngestModule()])

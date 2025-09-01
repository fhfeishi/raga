# container.py
from __future__ import annotations
from injector import Injector, Module, provider, singleton

from private_gpt.components.embedding.embedding_component import EmbeddingComponent
from private_gpt.components.llm.llm_component import LLMComponent
from private_gpt.components.node_store.node_store_component import NodeStoreComponent
from private_gpt.components.vector_store.vector_store_component import VectorStoreComponent

from .service import IngestService


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
        return IngestService(llm, vs, emb, ns)


def build_injector() -> Injector:
    return Injector([IngestModule()])

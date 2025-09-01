# private_gpt_method/ingest.py
from __future__ import annotations
import logging
import tempfile
from pathlib import Path
from typing import AnyStr, BinaryIO, List, Tuple, TYPE_CHECKING

from injector import inject, singleton
from llama_index.core.storage import StorageContext
from llama_index.core.storage.kvstore.simple_kvstore import SimpleKVStore

from .di import EmbeddingComponent, VectorStoreComponent, NodeStoreComponent
from .ingest_component import IngestionComponent
from .settings import settings

if TYPE_CHECKING:
    from llama_index.core.storage.docstore.types import RefDocInfo

from dataclasses import dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

@dataclass
class IngestedDoc:
    object: str
    doc_id: str
    doc_metadata: Optional[Dict[str, Any]] = None

    @staticmethod
    def curate_metadata(metadata: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        return dict(metadata) if metadata else None

    @staticmethod
    def from_document(document: Any) -> "IngestedDoc":
        doc_id = (
            getattr(document, "doc_id", None)
            or getattr(document, "node_id", None)
            or getattr(document, "id_", None)
            or document.metadata.get("doc_id")
            or "unknown-doc-id"
        )
        meta = getattr(document, "metadata", None)
        return IngestedDoc(object="ingest.document", doc_id=doc_id, doc_metadata=meta if isinstance(meta, dict) else None)

@singleton
class IngestService:
    @inject
    def __init__(
        self,
        embedding_component: EmbeddingComponent,
        vector_store_component: VectorStoreComponent,
        node_store_component: NodeStoreComponent,
    ) -> None:
        self._persist_dir = settings().storage.persist_dir

        # 组装 StorageContext
        self.storage_context = StorageContext.from_defaults(
            vector_store=vector_store_component.vector_store,
            docstore=node_store_component.doc_store,
            index_store=node_store_component.index_store,
            # 简单 KVStore，用于 LlamaIndex 的内部元数据
            kvstore=SimpleKVStore(),
        )

        # 句窗 or 句子分割
        ing = settings().ingest
        if ing.use_sentence_window:
            from llama_index.core.node_parser import SentenceWindowNodeParser
            node_parser = SentenceWindowNodeParser.from_defaults(window_size=ing.sentence_window_size)
        else:
            from llama_index.core.node_parser import SentenceSplitter
            node_parser = SentenceSplitter.from_defaults(
                chunk_size=ing.chunk_size,
                chunk_overlap=ing.chunk_overlap,
            )

        self.ingest_component = IngestionComponent(
            storage_context=self.storage_context,
            embed_model=embedding_component.embedding_model,
            node_parser=node_parser,
        )

    # —— APIs —— #
    def ingest_file(self, file_name: str, file_path: Path) -> List[IngestedDoc]:
        logger.info("Ingesting file_name=%s", file_name)
        documents = self.ingest_component.ingest(file_name, file_path)
        # 持久化（docstore/index_store 元数据）
        self.storage_context.persist(self._persist_dir)
        logger.info("Finished ingestion file_name=%s", file_name)
        return [IngestedDoc.from_document(doc) for doc in documents]

    def ingest_text(self, file_name: str, text: str) -> List[IngestedDoc]:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            try:
                path_to_tmp = Path(tmp.name)
                path_to_tmp.write_text(str(text))
                return self.ingest_file(file_name, path_to_tmp)
            finally:
                path_to_tmp.unlink(missing_ok=True)

    def ingest_bin_data(self, file_name: str, raw_file_data: BinaryIO) -> List[IngestedDoc]:
        data = raw_file_data.read()
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            try:
                path_to_tmp = Path(tmp.name)
                path_to_tmp.write_bytes(data)
                return self.ingest_file(file_name, path_to_tmp)
            finally:
                path_to_tmp.unlink(missing_ok=True)

    def bulk_ingest(self, files: List[Tuple[str, Path]]) -> List[IngestedDoc]:
        logger.info("Ingesting file_names=%s", [f[0] for f in files])
        docs = self.ingest_component.bulk_ingest(files)
        self.storage_context.persist(self._persist_dir)
        logger.info("Finished ingestion file_names=%s", [f[0] for f in files])
        return [IngestedDoc.from_document(d) for d in docs]

    def list_ingested(self) -> List[IngestedDoc]:
        # 读取 docstore 的 ref docs 信息
        ingested_docs: List[IngestedDoc] = []
        try:
            docstore = self.storage_context.docstore
            ref_docs: dict[str, RefDocInfo] | None = docstore.get_all_ref_doc_info()
            if not ref_docs:
                return ingested_docs
            for doc_id, ref_doc_info in ref_docs.items():
                doc_metadata = None
                if ref_doc_info is not None and ref_doc_info.metadata is not None:
                    doc_metadata = IngestedDoc.curate_metadata(ref_doc_info.metadata)
                ingested_docs.append(IngestedDoc(object="ingest.document", doc_id=doc_id, doc_metadata=doc_metadata))
        except ValueError:
            logger.warning("Got an exception when getting list of docs", exc_info=True)
        return ingested_docs

    def delete(self, doc_id: str) -> None:
        self.ingest_component.delete(doc_id)
        self.storage_context.persist(self._persist_dir)

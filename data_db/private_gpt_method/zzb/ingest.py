# private_gpt_method/ingest.py
from __future__ import annotations
import logging
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, AnyStr, BinaryIO, List, Tuple

from injector import inject, singleton
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.storage import StorageContext

# 复用原组件（保留工程师风格与行为）
from private_gpt.components.embedding.embedding_component import EmbeddingComponent
from private_gpt.components.llm.llm_component import LLMComponent
from private_gpt.components.node_store.node_store_component import NodeStoreComponent
from private_gpt.components.vector_store.vector_store_component import VectorStoreComponent

# 改为调用我们自己的适配层 & settings
from .ingest_component import get_ingestion_component
from .settings import settings

if TYPE_CHECKING:
    from llama_index.core.storage.docstore.types import RefDocInfo

# 为了避免强耦合 server 层，这里定义一个轻量的 IngestedDoc。
# 若你仍想用原 server 的 IngestedDoc，可改成：from private_gpt.server.ingest.model import IngestedDoc
from dataclasses import dataclass
from typing import Any, Dict, Optional

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
        doc_id = getattr(document, "doc_id", None) or getattr(document, "node_id", None) or getattr(document, "id_", None) or "unknown-doc-id"
        meta = getattr(document, "metadata", None)
        return IngestedDoc(object="ingest.document", doc_id=doc_id, doc_metadata=meta if isinstance(meta, dict) else None)


logger = logging.getLogger(__name__)


@singleton
class IngestService:
    @inject
    def __init__(
        self,
        llm_component: LLMComponent,
        vector_store_component: VectorStoreComponent,
        embedding_component: EmbeddingComponent,
        node_store_component: NodeStoreComponent,
    ) -> None:
        self.llm_service = llm_component  # 保留注入（即便此处暂未直接用）
        self.storage_context = StorageContext.from_defaults(
            vector_store=vector_store_component.vector_store,
            docstore=node_store_component.doc_store,
            index_store=node_store_component.index_store,
        )

        node_parser = SentenceWindowNodeParser.from_defaults()
        self.ingest_component = get_ingestion_component(
            self.storage_context,
            embed_model=embedding_component.embedding_model,
            transformations=[node_parser, embedding_component.embedding_model],
            settings=settings(),
        )

    # —— 对外 API —— #
    def ingest_file(self, file_name: str, file_data: Path) -> List[IngestedDoc]:
        logger.info("Ingesting file_name=%s", file_name)
        documents = self.ingest_component.ingest(file_name, file_data)
        logger.info("Finished ingestion file_name=%s", file_name)
        return [IngestedDoc.from_document(document) for document in documents]

    def ingest_text(self, file_name: str, text: str) -> List[IngestedDoc]:
        logger.debug("Ingesting text data with file_name=%s", file_name)
        return self._ingest_data(file_name, text)

    def ingest_bin_data(self, file_name: str, raw_file_data: BinaryIO) -> List[IngestedDoc]:
        logger.debug("Ingesting binary data with file_name=%s", file_name)
        file_data = raw_file_data.read()
        return self._ingest_data(file_name, file_data)

    def bulk_ingest(self, files: List[Tuple[str, Path]]) -> List[IngestedDoc]:
        logger.info("Ingesting file_names=%s", [f[0] for f in files])
        documents = self.ingest_component.bulk_ingest(files)
        logger.info("Finished ingestion file_name=%s", [f[0] for f in files])
        return [IngestedDoc.from_document(document) for document in documents]

    def list_ingested(self) -> List[IngestedDoc]:
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
                ingested_docs.append(
                    IngestedDoc(
                        object="ingest.document",
                        doc_id=doc_id,
                        doc_metadata=doc_metadata,
                    )
                )
        except ValueError:
            logger.warning("Got an exception when getting list of docs", exc_info=True)
        logger.debug("Found count=%s ingested documents", len(ingested_docs))
        return ingested_docs

    def delete(self, doc_id: str) -> None:
        """删除已入库文档。"""
        logger.info("Deleting the ingested document=%s in the doc and index store", doc_id)
        self.ingest_component.delete(doc_id)

    # —— 内部 —— #
    def _ingest_data(self, file_name: str, file_data: AnyStr) -> List[IngestedDoc]:
        logger.debug("Got file data of size=%s to ingest", len(file_data))
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            try:
                path_to_tmp = Path(tmp.name)
                if isinstance(file_data, bytes):
                    path_to_tmp.write_bytes(file_data)
                else:
                    path_to_tmp.write_text(str(file_data))
                return self.ingest_file(file_name, path_to_tmp)
            finally:
                tmp.close()
                path_to_tmp.unlink(missing_ok=True)

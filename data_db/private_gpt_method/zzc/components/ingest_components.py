# components/ingest_components.py


import itertools
import logging
import multiprocessing
import threading
from pathlib import Path
from typing import Any, Iterable, List, Tuple

from llama_index.core.ingestion import run_transformations
from llama_index.core.indices import VectorStoreIndex, load_index_from_storage
from llama_index.core.indices.base import BaseIndex
from llama_index.core.schema import BaseNode, Document, TransformComponent
from llama_index.core.storage import StorageContext

from settings import settings
from .ingest_helper import IngestionHelper

logger = logging.getLogger(__name__)


class BaseIngestor:
    def __init__(
        self,
        storage_context: StorageContext,
        embed_model: Any,
        transformations: List[TransformComponent],
    ) -> None:
        self.storage_context = storage_context
        self.embed_model = embed_model
        self.transformations = transformations
        self._index_lock = threading.Lock()
        self._index = self._init_index()

    def _init_index(self) -> BaseIndex:
        persist_dir = settings.paths.persist_dir / "li"
        persist_dir.mkdir(parents=True, exist_ok=True)
        # 尝试加载已存在的索引；若失败则新建空索引
        try:
            idx = load_index_from_storage(
                storage_context=self.storage_context,
                store_nodes_override=True,
                show_progress=True,
                embed_model=self.embed_model,
                transformations=self.transformations,
            )
        except Exception:
            logger.info("No existing index found, creating a new VectorStoreIndex...")
            idx = VectorStoreIndex.from_documents(
                [],
                storage_context=self.storage_context,
                store_nodes_override=True,
                show_progress=True,
                embed_model=self.embed_model,
                transformations=self.transformations,
            )
            idx.storage_context.persist(persist_dir=persist_dir)
        return idx

    def _persist(self) -> None:
        persist_dir = settings.paths.persist_dir / "li"
        self._index.storage_context.persist(persist_dir=persist_dir)

    # --- API ---
    def ingest(self, file_name: str, file_path: Path) -> List[Document]:
        raise NotImplementedError

    def bulk_ingest(self, files: List[Tuple[str, Path]]) -> List[Document]:
        raise NotImplementedError

    def delete(self, doc_id: str) -> None:
        with self._index_lock:
            self._index.delete_ref_doc(doc_id, delete_from_docstore=True)
            self._persist()


class SimpleIngestor(BaseIngestor):
    def ingest(self, file_name: str, file_path: Path) -> List[Document]:
        docs = IngestionHelper.from_file(file_name, file_path)
        with self._index_lock:
            for d in docs:
                self._index.insert(d, show_progress=True)
            self._persist()
        return docs

    def bulk_ingest(self, files: List[Tuple[str, Path]]) -> List[Document]:
        saved: List[Document] = []
        for name, path in files:
            saved.extend(self.ingest(name, path))
        return saved


class BatchIngestor(BaseIngestor):
    def __init__(self, storage_context: StorageContext, embed_model: Any, transformations: List[TransformComponent], count_workers: int = 2) -> None:
        super().__init__(storage_context, embed_model, transformations)
        assert len(self.transformations) >= 2, "Embeddings must be in the transformations"
        self.count_workers = max(1, int(count_workers))
        self._mp_pool = multiprocessing.Pool(processes=self.count_workers)

    def ingest(self, file_name: str, file_path: Path) -> List[Document]:
        docs = IngestionHelper.from_file(file_name, file_path)
        return self._save_docs(docs)

    def bulk_ingest(self, files: List[Tuple[str, Path]]) -> List[Document]:
        docs_nested = self._mp_pool.starmap(IngestionHelper.from_file, files)
        docs = list(itertools.chain.from_iterable(docs_nested))
        return self._save_docs(docs)

    def _save_docs(self, documents: List[Document]) -> List[Document]:
        nodes = run_transformations(documents, self.transformations, show_progress=True)
        with self._index_lock:
            self._index.insert_nodes(nodes, show_progress=True)
            for d in documents:
                self._index.docstore.set_document_hash(d.get_doc_id(), d.hash)
            self._persist()
        return documents


def build_ingestor(storage_context: StorageContext, embed_model: Any, transformations: List[TransformComponent]):
    """根据（可选的）settings.embedding.ingest_mode 构建 ingestor，默认为 simple。
    为了与当前 settings.py 兼容，这里做了“属性存在检查”，没有该字段时回落 simple。
    """
    mode = getattr(getattr(settings, "embedding_model", object()), "ingest_mode", None)
    workers = getattr(getattr(settings, "embedding_model", object()), "count_workers", 2)
    if mode == "batch":
        return BatchIngestor(storage_context, embed_model, transformations, workers)
    return SimpleIngestor(storage_context, embed_model, transformations)
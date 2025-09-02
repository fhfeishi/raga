# api/ingest_service.py



from __future__ import annotations

import io
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from llama_index.core.storage import StorageContext

from settings import settings
from components.ingest_components import build_ingestor

logger = logging.getLogger(__name__)


@dataclass
class IngestedDoc:
    object: str
    doc_id: str
    doc_metadata: dict

    @staticmethod
    def curate_metadata(md: dict) -> dict:
        md = dict(md or {})
        for k in ("doc_id", "window", "original_text"):
            md.pop(k, None)
        return md


class IngestService:
    """最小可用 Ingest 服务，不绑定具体 Web 框架。
    API 层（FastAPI/Flask）只需把 UploadFile 或文本交给它。
    """

    def __init__(self) -> None:
        # 确保 LlamaIndex 全局设置
        settings.configure_llama_index()

        # 组装向量库 + StorageContext
        vstore = settings.build_vector_store()
        li_persist = settings.paths.persist_dir / "li"
        li_persist.mkdir(parents=True, exist_ok=True)
        self.storage_ctx = StorageContext.from_defaults(vector_store=vstore, persist_dir=str(li_persist))

        # 构建变换链：splitter -> embed
        splitter = settings.build_text_splitter()
        embed = settings.build_embedding()
        self.ingestor = build_ingestor(self.storage_ctx, embed, [splitter, embed])

    # ------- APIs -------
    def ingest_file(self, file_name: str, file_path: Path) -> List[IngestedDoc]:
        docs = self.ingestor.ingest(file_name, file_path)
        return [IngestedDoc(object="ingest.document", doc_id=d.doc_id, doc_metadata=IngestedDoc.curate_metadata(d.metadata)) for d in docs]

    def ingest_text(self, file_name: str, text: str) -> List[IngestedDoc]:
        # 写成临时文件再走统一流程（复用 file readers 的路径判断）
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
            path = Path(tmp.name)
            path.write_text(text, encoding="utf-8")
        try:
            return self.ingest_file(file_name, path)
        finally:
            try:
                path.unlink(missing_ok=True)  # type: ignore[arg-type]
            except Exception:
                pass

    def bulk_ingest(self, files: List[Tuple[str, Path]]):
        docs = self.ingestor.bulk_ingest(files)
        return [IngestedDoc(object="ingest.document", doc_id=d.doc_id, doc_metadata=IngestedDoc.curate_metadata(d.metadata)) for d in docs]

    def list_ingested(self) -> List[IngestedDoc]:
        out: List[IngestedDoc] = []
        try:
            ref_docs = self.storage_ctx.docstore.get_all_ref_doc_info()  # type: ignore[attr-defined]
            if not ref_docs:
                return out
            for doc_id, ref_info in ref_docs.items():
                md = None
                if ref_info is not None and getattr(ref_info, "metadata", None) is not None:
                    md = IngestedDoc.curate_metadata(ref_info.metadata)
                out.append(IngestedDoc(object="ingest.document", doc_id=doc_id, doc_metadata=md or {}))
        except Exception:
            logger.exception("listing ingested docs failed")
        return out

    def delete(self, doc_id: str) -> None:
        self.ingestor.delete(doc_id)
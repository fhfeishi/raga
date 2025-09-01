# models.py


from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict

try:
    # 如果你仍保留原工程的实现，可以直接复用
    from private_gpt.server.ingest.model import IngestedDoc as _OriginalIngestedDoc
    IngestedDoc = _OriginalIngestedDoc  # 直接别名到原类型
except Exception:
    # 轻量实现（与原行为保持兼容）
    @dataclass
    class IngestedDoc:
        object: str
        doc_id: str
        doc_metadata: Dict[str, Any] | None = None

        @staticmethod
        def curate_metadata(metadata: Dict[str, Any] | None) -> Dict[str, Any] | None:
            if metadata is None:
                return None
            # 这里不过度“裁剪”，只做一个安全 copy。若有需要可按项目约定过滤字段。
            return dict(metadata)

        @staticmethod
        def from_document(document: Any) -> "IngestedDoc":
            # 兼容 llama-index 的多种 Document/BaseNode 结构
            doc_id = getattr(document, "doc_id", None) or getattr(document, "node_id", None)
            if doc_id is None and hasattr(document, "metadata"):
                doc_id = document.metadata.get("doc_id") or document.metadata.get("node_id")
            if doc_id is None:
                # 【需留意】极端情况下无法取到 id，这里兜底
                doc_id = getattr(document, "id_", None) or "unknown-doc-id"

            metadata = None
            if hasattr(document, "metadata"):
                metadata = IngestedDoc.curate_metadata(document.metadata)
            return IngestedDoc(object="ingest.document", doc_id=doc_id, doc_metadata=metadata)

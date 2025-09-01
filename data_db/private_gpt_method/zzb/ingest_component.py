# private_gpt_method/ingest_component.py
from __future__ import annotations
from typing import Any, List

try:
    # 优先复用原实现（推荐）
    from private_gpt.components.ingest.ingest_component import (
        get_ingestion_component as _orig_get_ingestion_component,
    )

    def get_ingestion_component(
        storage_context: Any,
        embed_model: Any,
        transformations: List[Any],
        settings: Any,
    ) -> Any:
        return _orig_get_ingestion_component(
            storage_context,
            embed_model=embed_model,
            transformations=transformations,
            settings=settings,
        )

except Exception:
    # —— 降级：提供一个极简 Ingestor（示意，需要替换为你的实际策略）——
    from dataclasses import dataclass
    from pathlib import Path
    from llama_index.core import VectorStoreIndex, Document

    @dataclass
    class _SimpleIngestComponent:
        storage_context: Any
        embed_model: Any
        transformations: List[Any]

        def ingest(self, file_name: str, file_path: Path):
            # 简化版：读取纯文本（示意，需要替换）
            text = file_path.read_text(encoding="utf-8", errors="ignore")
            doc = Document(text=text, metadata={"file_name": file_name})
            # 直接用 VectorStoreIndex 构建；真实项目应走 transformations 分块等
            VectorStoreIndex.from_documents([doc], storage_context=self.storage_context)
            return [doc]

        def bulk_ingest(self, files: list[tuple[str, Path]]):
            docs = []
            for name, path in files:
                docs.extend(self.ingest(name, path))
            return docs

        def delete(self, doc_id: str):
            # （示意，需要替换）实际应访问 docstore/index_store 删除
            raise NotImplementedError("请在有需要时对接你的 docstore 删除逻辑。")

    def get_ingestion_component(
        storage_context: Any,
        embed_model: Any,
        transformations: List[Any],
        settings: Any,
    ) -> Any:
        return _SimpleIngestComponent(
            storage_context=storage_context,
            embed_model=embed_model,
            transformations=transformations,
        )

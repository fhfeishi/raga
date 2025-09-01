# private_gpt_method/ingest_component.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List

from llama_index.core import VectorStoreIndex, Document, StorageContext, Settings as LISettings
from llama_index.core.node_parser import SentenceWindowNodeParser, SentenceSplitter

@dataclass
class IngestionComponent:
    storage_context: StorageContext
    embed_model: Any
    node_parser: Any  # 允许传任意 parser（SentenceWindow 或 Splitter）

    def _read_file(self, file_name: str, file_path: Path) -> List[Document]:
        # 简化读取：单文件 -> Document
        # 如需更复杂（PDF 分页、图片 OCR），后续再在这里扩展
        text = file_path.read_text(encoding="utf-8", errors="ignore")
        return [Document(text=text, metadata={"file_name": file_name, "doc_id": file_name})]

    def _index_docs(self, docs: List[Document]) -> List[Document]:
        # 应用分块/句窗 → 构建向量索引（嵌入由 embed_model 完成）
        # LlamaIndex 全局设置 embed_model
        old_embed = LISettings.embed_model
        LISettings.embed_model = self.embed_model
        try:
            # 句窗解析器直接将文档转为节点；SentenceSplitter 也是类似
            nodes = self.node_parser.get_nodes_from_documents(docs)
            VectorStoreIndex(
                nodes=nodes,
                storage_context=self.storage_context,
                embed_model=self.embed_model,
            )
            return docs
        finally:
            LISettings.embed_model = old_embed

    def ingest(self, file_name: str, file_path: Path) -> List[Document]:
        docs = self._read_file(file_name, file_path)
        return self._index_docs(docs)

    def bulk_ingest(self, files: Iterable[tuple[str, Path]]) -> List[Document]:
        out: List[Document] = []
        for fname, fpath in files:
            out.extend(self.ingest(fname, fpath))
        return out

    def delete(self, doc_id: str) -> None:
        # 通过 ref_doc_id 删除（ChromaVectorStore 支持）
        # 注意：前提是入库时将 nodes 的 ref_doc_id 设置为 file_name / doc_id
        # 上面我们在 metadata 里放了 doc_id=file_name，但
        # VectorStoreIndex 默认会把 source 节点的 ref_doc_id 设为 doc.doc_id（如果存在）。
        # 因此我们在 _read_file 里显式设置了 metadata["doc_id"]=file_name。
        # llama_index 的 ChromaVectorStore.delete(ref_doc_id=...) 会根据元数据删。
        self.storage_context.vector_store.delete(ref_doc_id=doc_id)

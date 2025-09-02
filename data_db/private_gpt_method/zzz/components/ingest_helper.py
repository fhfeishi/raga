
# components/ingest_helper.py

import logging
from pathlib import Path
from typing import Any, Dict

from llama_index.core.schema import Document

# 复用 settings 中的柔性导入
from settings import settings

logger = logging.getLogger(__name__)


def _try_loading_included_file_formats() -> Dict[str, type]:
    """尝试加载可选的 llama-index file readers（跨版本路径兼容）。
    需要安装 `llama-index-readers-file` 才能覆盖大多数格式。
    """
    try:
        DocxReader = settings._import([
            "llama_index.readers.file.docs.DocxReader",
            "llama_index.readers.file.docs.DocxReader",
        ])
        HWPReader = settings._import([
            "llama_index.readers.file.docs.HWPReader",
            "llama_index.readers.file.docs.HWPReader",
        ])
        PDFReader = settings._import([
            "llama_index.readers.file.docs.PDFReader",
            "llama_index.readers.file.docs.PDFReader",
        ])
        EpubReader = settings._import([
            "llama_index.readers.file.epub.EpubReader",
            "llama_index.readers.file.epub.EpubReader",
        ])
        ImageReader = settings._import([
            "llama_index.readers.file.image.ImageReader",
            "llama_index.readers.file.image.ImageReader",
        ])
        IPYNBReader = settings._import([
            "llama_index.readers.file.ipynb.IPYNBReader",
            "llama_index.readers.file.ipynb.IPYNBReader",
        ])
        MarkdownReader = settings._import([
            "llama_index.readers.file.markdown.MarkdownReader",
            "llama_index.readers.file.markdown.MarkdownReader",
        ])
        MboxReader = settings._import([
            "llama_index.readers.file.mbox.MboxReader",
            "llama_index.readers.file.mbox.MboxReader",
        ])
        PptxReader = settings._import([
            "llama_index.readers.file.slides.PptxReader",
            "llama_index.readers.file.slides.PptxReader",
        ])
        PandasCSVReader = settings._import([
            "llama_index.readers.file.tabular.PandasCSVReader",
            "llama_index.readers.file.tabular.PandasCSVReader",
        ])
        VideoAudioReader = settings._import([
            "llama_index.readers.file.video_audio.VideoAudioReader",
            "llama_index.readers.file.video_audio.VideoAudioReader",
        ])
        JSONReader = settings._import([
            "llama_index.core.readers.json.JSONReader",
            "llama_index.readers.json.JSONReader",
        ])
    except Exception:
        # 未安装可选 readers 包时，返回精简集（后续会回落到纯文本）
        return {}

    return {
        ".hwp": HWPReader,
        ".pdf": PDFReader,
        ".docx": DocxReader,
        ".pptx": PptxReader,
        ".ppt": PptxReader,
        ".pptm": PptxReader,
        ".jpg": ImageReader,
        ".png": ImageReader,
        ".jpeg": ImageReader,
        ".mp3": VideoAudioReader,
        ".mp4": VideoAudioReader,
        ".csv": PandasCSVReader,
        ".epub": EpubReader,
        ".md": MarkdownReader,
        ".mbox": MboxReader,
        ".ipynb": IPYNBReader,
        ".json": JSONReader,
    }


FILE_READER_CLS = _try_loading_included_file_formats()


class IngestionHelper:
    """把文件/文本转换为 LlamaIndex Documents，并规范元数据。
    与 private-gpt 的思路一致：尽量多格式 -> Document 列表 -> 清洗/抽取元信息。
    """

    @staticmethod
    def from_text(file_name: str, text: str) -> list[Document]:
        # 统一成 Document 接口
        return IngestionHelper._postprocess([Document(text=text, metadata={"file_name": file_name})])

    @staticmethod
    def from_file(file_name: str, file_path: Path) -> list[Document]:
        ext = file_path.suffix.lower()
        reader_cls = FILE_READER_CLS.get(ext)
        if reader_cls is None:
            # 回落为纯文本读取
            try:
                content = file_path.read_text(encoding="utf-8")
            except Exception:
                # 二进制或未知编码，直接读取 bytes 再解码忽略错误
                content = file_path.read_bytes().decode("utf-8", errors="ignore")
            docs = [Document(text=content, metadata={"file_name": file_name})]
            return IngestionHelper._postprocess(docs)

        # 专用 reader
        reader = reader_cls()
        docs = reader.load_data(file_path)
        for d in docs:
            d.metadata["file_name"] = file_name
        return IngestionHelper._postprocess(docs)

    @staticmethod
    def _postprocess(documents: list[Document]) -> list[Document]:
        # 清理 NUL 字符并整理 metadata/排除项
        for d in documents:
            if d.text:
                d.text = d.text.replace("\u0000", "")
            d.metadata["doc_id"] = d.doc_id
            d.excluded_embed_metadata_keys = ["doc_id"]
            d.excluded_llm_metadata_keys = ["file_name", "doc_id", "page_label"]
        return documents

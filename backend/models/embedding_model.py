# models/enbedding_model.py

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaEmbeddings
from injector import singleton

import logging
logger = logging.getLogger(__name__)


enbedding_model_config = {
    "type": "huggingface", # "huggingface" 或 "ollama"
    "huggingface_model": "Qwen/Qwen3-Embedding-0.6B",
    "ollama_model": "nomic-embed-text",
    "ollama_base_url": "http://localhost:11434",
    "device": "cuda" if __import__("os").environ.get("CUDA_VISIBLE_DEVICES") else "cpu", 
}

@singleton
def enbedding_model_cratefn(emb_cfg=enbedding_model_config):
    if emb_cfg["type"] == "huggingface":
        logger.info(f"🔧 使用 HuggingFace 嵌入模型: {emb_cfg['huggingface_model']}")
        return HuggingFaceEmbeddings(
            model_name=emb_cfg["huggingface_model"],
            model_kwargs={"device": emb_cfg["device"]}
        )
    elif emb_cfg["type"] == "ollama":
        logger.info(f"🔧 使用 Ollama 嵌入模型: {emb_cfg['ollama_model']}")
        return OllamaEmbeddings(
            model=emb_cfg["ollama_model"],
            base_url=emb_cfg["ollama_base_url"]
        )
    else:
        raise ValueError(f"不支持的嵌入类型: {emb_cfg['type']}")
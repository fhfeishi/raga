# settings.py
from __future__ import annotations
import os
from dataclasses import dataclass, field

def _bool(s: str | None, default: bool = False) -> bool:
    if s is None:
        return default
    return s.lower() in {"1", "true", "yes", "on"}

@dataclass
class HFSettings:
    model_name_or_path: str = os.getenv("PGM_EMBED_MODEL", "nomic-ai/nomic-embed-text-v1.5")
    device: str = os.getenv("PGM_DEVICE", "cpu")
    local_files_only: bool = _bool(os.getenv("PGM_LOCAL_FILES_ONLY", "1"), True)
    trust_remote_code: bool = _bool(os.getenv("PGM_TRUST_REMOTE_CODE", "1"), True)
    cache_dir: str | None =  os.getenv("HF_HOME") 

@dataclass
class IngestSettings:
    chunk_size: int = int(os.getenv("PGM_CHUNK_SIZE", "1024"))
    chunk_overlap: int = int(os.getenv("PGM_CHUNK_OVERLAP", "128"))
    sentence_window_size: int = int(os.getenv("PGM_SENT_WINDOW", "3"))
    use_sentence_window: bool = _bool(os.getenv("PGM_USE_SENT_WINDOW", "1"), True)

@dataclass
class VectorStoreSettings:
    backend: str = os.getenv("PGM_VECTOR_BACKEND", "chroma")
    chroma_dir: str = os.getenv("PGM_CHROMA_DIR", "local_data/private_gpt_method/chroma")
    collection: str = os.getenv("PGM_CHROMA_COLLECTION", "pgm_collection")

@dataclass
class StorageSettings:
    persist_dir: str = os.getenv("PGM_PERSIST_DIR", "local_data/private_gpt_method/storage")

@dataclass
class Settings:
    # ✅ 必须用 default_factory，不能直接写 HFSettings()
    hf: HFSettings = field(default_factory=HFSettings)
    ingest: IngestSettings = field(default_factory=IngestSettings)
    vs: VectorStoreSettings = field(default_factory=VectorStoreSettings)
    storage: StorageSettings = field(default_factory=StorageSettings)

def settings() -> Settings:
    return Settings()

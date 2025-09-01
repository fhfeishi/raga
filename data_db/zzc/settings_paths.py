# private_gpt_method/settings_paths.py
from __future__ import annotations
from pathlib import Path
from .settings import settings

def ensure_dirs() -> None:
    s = settings()
    Path(s.vs.chroma_dir).mkdir(parents=True, exist_ok=True)
    Path(s.storage.persist_dir).mkdir(parents=True, exist_ok=True)

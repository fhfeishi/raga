# private_gpt_method/settings.py
from __future__ import annotations

try:
    from private_gpt.settings.settings import settings as _orig_settings

    def settings():
        """转发到原工程 settings()。后续我们再迁移关键项到本包。"""
        return _orig_settings()
except Exception:
    # —— 降级占位（示意，需要替换）——
    from dataclasses import dataclass

    @dataclass
    class _Settings:
        # 这里补必需字段（示意，需要替换/补全）
        chunk_size: int = 1024
        chunk_overlap: int = 128

    def settings():
        return _Settings()

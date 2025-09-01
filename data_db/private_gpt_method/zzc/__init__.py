# private_gpt_method/__init__.py
from .di import build_injector
from .ingest import IngestService

__all__ = ["build_injector", "IngestService"]

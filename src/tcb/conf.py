"""Fixed pipeline constants (NOT env-configurable) + env Settings for the LLM/runtime."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# ---- fixed: changing any of these invalidates the shipped index (see manifest) ----
DOC_ID = "tcb-ar-2025"
EMBED_MODEL = "BAAI/bge-m3"
EMBED_REV = "5617a9f61b028005a4858fdac845db406aefb181"   # HF main commit seen 10 Oct 2026; re-verified at download
EMBED_DIM = 1024
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
RERANK_REV = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
INDEX_FORMAT = 1
TOKENIZER = "tcb-vi-v1"
CHUNK_MAX_WORDS = 220        # narrative chunk budget (syllable-words)
TABLE_ROWS_PER_CHUNK = 12
BM25_K1, BM25_B = 1.5, 0.75  # bm25s defaults, recorded in manifest
TOP_BM25 = TOP_DENSE = 40    # E3/E4 may revise before M3 freeze
RRF_K = 60                   # E3b
MODELS_DIR = Path("/models")  # baked into the image (Dockerfile); override only by editing code


@dataclass(frozen=True)
class Settings:
    """Env-driven, non-retrieval settings (LLM provider etc. are read in llm/)."""
    index_dir: Path
    host: str
    port: int
    trace_dir: Path
    log_level: str


def settings() -> Settings:
    return Settings(
        index_dir=Path(os.getenv("INDEX_DIR", "idx")),
        host=os.getenv("APP_HOST", "127.0.0.1"),
        port=int(os.getenv("APP_PORT", "8000")),
        trace_dir=Path(os.getenv("TRACE_DIR", "out/trace")),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
    )

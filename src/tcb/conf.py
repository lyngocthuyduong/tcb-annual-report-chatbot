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
INDEX_FORMAT = 1
TOKENIZER = "tcb-vi-v2"   # v2: canonical decimals drop trailing zeros (16,0 -> 16)
CHUNK_MAX_WORDS = 220        # narrative chunk budget (syllable-words)
TABLE_ROWS_PER_CHUNK = 12
NODE_CONTEXT = "chapter>heading;flow2"   # node text starts with [chapter › page title › subtitle]; flow2 = column order + run-on sentences joined
BM25_K1, BM25_B = 1.5, 0.75  # bm25s defaults, recorded in manifest
TOP_BM25 = TOP_DENSE = 40
RRF_K = 60                   # E3b
def _models_dir() -> Path:
    """Where the two fixed models live. Docker: /models (baked). Local: $TCB_MODELS_DIR or <repo>/models.
    This changes only the LOCATION; model identity/revision stay fixed above and are checked via <dir>/<model>/.rev."""
    env = os.getenv("TCB_MODELS_DIR")
    if env:
        return Path(env)
    return Path("/models") if Path("/models").exists() else Path(__file__).resolve().parents[2] / "models"


MODELS_DIR = _models_dir()
PDF_SHA256 = "2db3dfe6e7a05f889ec2e93bf16dda4b6a7f3334df440b7cb0761df2e89344d9"   # data/ar.pdf
POPPLER_HINT = ("Install poppler and put it on PATH (or set POPPLER_BIN to its bin folder): "
                "Windows: `conda install -c conda-forge poppler` or https://github.com/oschwartz10612/poppler-windows/releases ; "
                "macOS: `brew install poppler` ; Ubuntu/Debian: `sudo apt install poppler-utils`")


def poppler(name: str) -> str:
    """Absolute path of a poppler tool (pdftotext, pdfimages) or a clear error."""
    import shutil
    b = os.getenv("POPPLER_BIN")
    p = shutil.which(name, path=b) if b else None
    p = p or shutil.which(name)
    if not p:
        raise RuntimeError(f"{name} not found. {POPPLER_HINT}")
    return p


def load_env(path: str | Path = ".env") -> list[str]:
    """Local (non-Docker) runs: read KEY=VALUE lines from .env into os.environ WITHOUT overriding real env vars.
    Docker Compose does this itself via env_file. Returns the keys that were set (never their values)."""
    p = Path(path)
    if not p.exists():
        return []
    done = []
    for line in p.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip()
        if v[:1] in "\"'" and v[-1:] == v[:1] and len(v) >= 2:
            v = v[1:-1]
        else:
            v = v.split(" #", 1)[0].strip()
        if k and v and k not in os.environ:
            os.environ[k] = v; done.append(k)
    return done


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

# Top fused blocks given to the LLM (E1..E10). Run m4 (dev+val, 85 answerable): recall 0.882 at 10; a cross-encoder over 20-40
# candidates reached 0.894 (one item) at ~2 s per candidate on 2 CPU cores, so there is no reranker (DESIGN 17.1).
FINAL_EVIDENCE = 15          # E4 run m5 (no reranker): recall 0.871 at 10, 0.929 at 15 and 20 -> smallest k within 2 pts
# $/1M tokens (in, out) -- public list prices checked 10 Oct 2026; used only for the cost estimate in traces
PRICES = {"gpt-4o-mini": (0.15, 0.60), "claude-sonnet-5-5": (2.0, 10.0),
          "claude-haiku-5-5": (0.10, 0.50)}   # Haiku: Anthropic price page, prompts up to 100k tokens (13 Oct 2026)
PROMPT_EVIDENCE_CHARS = 22000   # budget for evidence text in one prompt (~7k tokens @ ~3 chars/token; Ollama runs with num_ctx 12288)

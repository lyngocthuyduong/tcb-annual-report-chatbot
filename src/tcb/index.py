"""Load + validate the shipped index. Fails closed with an actionable message; never rebuilds."""
from __future__ import annotations

import json
from pathlib import Path

from . import conf
from .build import sha256

FIX = ("Re-fetch the repo (git pull / re-clone) so idx/ is intact. Rebuild only manually: "
       "docker compose run --rm app tcb build data/ar.pdf out/idx, then replace idx/ (see README).")


class IndexError_(RuntimeError):
    """Index missing / corrupt / incompatible. CLI exits with code 3."""

    def __init__(self, code: str, msg: str):
        super().__init__(f"[{code}] {msg}\n{FIX}")
        self.code = code


REQUIRED = ("index_format_version", "source", "embedding", "bm25", "counts", "files")


def validate(index_dir: str | Path, embed_model: str = conf.EMBED_MODEL, embed_rev: str = conf.EMBED_REV,
             check_snapshot: bool = False, strict_chunking: bool = True) -> dict:
    d = Path(index_dir)
    mf = d / "manifest.json"
    if not d.is_dir() or not mf.exists():
        raise IndexError_("missing", f"{mf} not found")
    try:
        m = json.loads(mf.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise IndexError_("corrupt", f"manifest.json unreadable: {e}")
    for k in REQUIRED:
        if k not in m: raise IndexError_("corrupt", f"manifest missing key '{k}'")
    if m["index_format_version"] != conf.INDEX_FORMAT:
        raise IndexError_("format", f"index format {m['index_format_version']} != code {conf.INDEX_FORMAT}")
    e = m["embedding"]
    if e.get("model") != embed_model or e.get("revision") != embed_rev:
        raise IndexError_("embedding", f"index built with {e.get('model')}@{e.get('revision')}, "
                                       f"code expects {embed_model}@{embed_rev}")
    ch = m.get("chunking", {})
    if strict_chunking and (ch.get("narrative_max_words") != conf.CHUNK_MAX_WORDS or ch.get("table_rows_per_chunk") != conf.TABLE_ROWS_PER_CHUNK):
        raise IndexError_("format", f"index chunking {ch.get('narrative_max_words')}/{ch.get('table_rows_per_chunk')} words/rows "
                                    f"!= code {conf.CHUNK_MAX_WORDS}/{conf.TABLE_ROWS_PER_CHUNK} (experimental index?)")
    if m["bm25"].get("tokenizer") != conf.TOKENIZER:
        raise IndexError_("tokenizer", f"index tokenizer {m['bm25'].get('tokenizer')} != {conf.TOKENIZER}")
    if check_snapshot:
        rev = conf.MODELS_DIR / "bge-m3" / ".rev"
        if not rev.exists() or rev.read_text(encoding="utf-8").strip() != embed_rev:
            raise IndexError_("snapshot", f"local bge-m3 snapshot revision != {embed_rev} (image/model mismatch)")
    # files: every listed file exists with the same sha256; no unlisted files
    listed = set(m["files"])
    present = {p.relative_to(d).as_posix() for p in d.rglob("*") if p.is_file() and p.name != "manifest.json"}
    if present - listed: raise IndexError_("corrupt", f"unlisted files: {sorted(present - listed)[:5]}")
    for rel, h in m["files"].items():
        p = d / rel
        if not p.exists(): raise IndexError_("missing", f"{rel} listed but absent")
        if sha256(p) != h: raise IndexError_("corrupt", f"sha256 mismatch: {rel}")
    # counts
    n = m["counts"]["nodes"]
    lines = sum(1 for _ in open(d / "nodes.jsonl", encoding="utf-8"))
    if lines != n: raise IndexError_("corrupt", f"nodes.jsonl has {lines} lines, manifest says {n}")
    ids = json.loads((d / "bm25" / "ids.json").read_text(encoding="utf-8"))
    if len(ids) != n: raise IndexError_("corrupt", f"bm25 ids {len(ids)} != {n}")
    ds = json.loads((d / "llamaindex" / "docstore.json").read_text(encoding="utf-8"))
    nd = len(ds.get("docstore/data", {}))
    if nd != n: raise IndexError_("corrupt", f"llamaindex docstore has {nd} nodes, manifest says {n}")
    return m


class Index:
    """Loaded index: node table, bm25s retriever, LlamaIndex vector index."""

    def __init__(self, manifest, nodes, bm25, ids, vector):
        self.manifest, self.nodes, self.bm25, self.ids, self.vector = manifest, nodes, bm25, ids, vector

    @classmethod
    def load(cls, index_dir: str | Path, embedder, **vkw) -> "Index":
        m = validate(index_dir, **vkw)
        d = Path(index_dir)
        nodes = {}
        with open(d / "nodes.jsonl", encoding="utf-8") as f:
            for ln in f:
                x = json.loads(ln); nodes[x["chunk_id"]] = x
        import bm25s
        from llama_index.core import StorageContext, load_index_from_storage
        bm = bm25s.BM25.load(str(d / "bm25"))
        ids = json.loads((d / "bm25" / "ids.json").read_text(encoding="utf-8"))
        vec = load_index_from_storage(StorageContext.from_defaults(persist_dir=str(d / "llamaindex")),
                                      embed_model=embedder)
        return cls(m, nodes, bm, ids, vec)

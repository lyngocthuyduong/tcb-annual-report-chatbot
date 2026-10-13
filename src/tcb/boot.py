"""Wire the runtime: provider config -> index validation (fail closed) -> real models. Tests inject stand-ins."""
from __future__ import annotations

import json
from pathlib import Path

from . import ask, conf, find, llm
from .index import Index, validate


def load_retriever(index_dir=None, embedder=None, lenient: bool = False) -> find.Retriever:
    """Index validation (fail closed) -> real models -> Retriever. No LLM involved (used by `tcb exp`).
    lenient=True (experiments only) loads an index built with other chunking; the production path never does."""
    d = Path(index_dir or conf.settings().index_dir)
    if embedder is None:
        from .build import real_embedder
        validate(d, strict_chunking=not lenient, pdf=conf.PDF_PATH)   # fail closed first (exit 3): hashes only, no models
        embedder = real_embedder()                     # ModelMissing (exit 5) if /models is absent
        ix = Index.load(d, embedder, check_snapshot=True, strict_chunking=not lenient)
    else:
        ix = Index.load(d, embedder, embed_model=ix_model(d), embed_rev=ix_rev(d), strict_chunking=not lenient)
    gloss = json.loads((d / "gloss.json").read_text(encoding="utf-8"))
    return find.Retriever(ix, gloss)


def load_engine(env=None, index_dir=None, embedder=None, provider=None) -> ask.Engine:
    """Order matters: cheap config errors first, then the index, then the (slow) model.
    embedder/provider are injectable only for tests; production passes none of them."""
    prov = provider or llm.get(env)
    rt = load_retriever(index_dir, embedder)
    return ask.Engine(rt, prov, llm.max_tokens(env))


def ix_model(d):  # injected-embedder path (tests): trust the manifest's own identity
    return json.loads((Path(d) / "manifest.json").read_text(encoding="utf-8"))["embedding"]["model"]


def ix_rev(d):
    return json.loads((Path(d) / "manifest.json").read_text(encoding="utf-8"))["embedding"]["revision"]

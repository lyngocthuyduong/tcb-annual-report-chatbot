"""Wire the runtime: provider config -> index validation (fail closed) -> real models. Tests inject stand-ins."""
from __future__ import annotations

import json
from pathlib import Path

from . import ask, conf, find, llm
from .index import Index, validate


def load_engine(env=None, index_dir=None, embedder=None, reranker=None, provider=None) -> ask.Engine:
    """Order matters: cheap config errors first, then the index, then the (slow) models.
    embedder/reranker/provider are injectable ONLY for tests; production passes none of them."""
    prov = provider or llm.get(env)
    d = Path(index_dir or conf.settings().index_dir)
    if embedder is None:
        from .build import real_embedder
        validate(d)                                    # fail closed first (exit 3), cheap: hashes only, no models
        embedder = real_embedder()                     # ModelMissing (exit 5) if /models is absent
        ix = Index.load(d, embedder, check_snapshot=True)
    else:
        ix = Index.load(d, embedder, embed_model=ix_model(d), embed_rev=ix_rev(d))
    gloss = json.loads((d / "gloss.json").read_text(encoding="utf-8"))
    rt = find.Retriever(ix, reranker or find.CrossEncoderReranker(), gloss)
    return ask.Engine(rt, prov, llm.max_tokens(env), conf.TAU_LOW)


def ix_model(d):  # injected-embedder path (tests): trust the manifest's own identity
    return json.loads((Path(d) / "manifest.json").read_text())["embedding"]["model"]


def ix_rev(d):
    return json.loads((Path(d) / "manifest.json").read_text())["embedding"]["revision"]

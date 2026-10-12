"""Hybrid retrieval: BM25 (tcb-vi-v1 tokens) + dense (LlamaIndex) -> RRF -> cross-encoder rerank.
Every stage's ranking is kept so eval can attribute a miss to the stage that lost the gold page."""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from typing import Protocol

from . import conf, vi
from .index import Index


class Reranker(Protocol):
    def score(self, query: str, texts: list[str]) -> list[float]: ...


class CrossEncoderReranker:
    """BAAI/bge-reranker-v2-m3 from the baked snapshot. Loaded lazily; never downloads."""

    def __init__(self, path=None, max_length: int = 512, device: str | None = None):
        self.path, self.max_length, self._m = path or conf.MODELS_DIR / "bge-reranker-v2-m3", max_length, None
        self.device = device

    def score(self, query, texts):
        if self._m is None:
            if not self.path.exists():
                raise FileNotFoundError(f"{self.path} missing: reranker is baked into the Docker image (never downloaded at run time)")
            from sentence_transformers import CrossEncoder
            self._m = CrossEncoder(str(self.path), device=self.device or os.getenv("TCB_DEVICE", "cpu"), max_length=self.max_length)   # cpu = what ships; cuda only for faster Colab runs
        return [float(x) for x in self._m.predict([(query, t) for t in texts], batch_size=8)]


_ABBR = re.compile(r"^[A-Za-zĐ0-9/ ]{2,10}$")


def expand(query: str, gloss: dict) -> str:
    """Append the report's own glossary definition after an abbreviation: 'CASA' -> 'CASA (Tiền gửi không kỳ hạn)'."""
    q = vi.nfc(query)
    extra = []
    for term, v in gloss.items():
        if _ABBR.match(term) and any(c.isupper() for c in term) and re.search(rf"(?<!\w){re.escape(term)}(?!\w)", q):
            extra.append(v["definition"][:100].strip())
    return q if not extra else f"{q} ({'; '.join(extra)})"


def rrf(rankings: list[list[str]], k: int = conf.RRF_K) -> list[tuple[str, float]]:
    s: dict[str, float] = {}
    for r in rankings:
        for i, cid in enumerate(r, 1):
            s[cid] = s.get(cid, 0.0) + 1.0 / (k + i)
    return sorted(s.items(), key=lambda x: (-x[1], x[0]))


@dataclass
class Result:
    query: str
    bm25: list[str] = field(default_factory=list)
    dense: list[str] = field(default_factory=list)
    fused: list[str] = field(default_factory=list)
    reranked: list[tuple[str, float]] = field(default_factory=list)   # (chunk_id, score) best first
    seconds: dict = field(default_factory=dict)

    @property
    def top_score(self) -> float | None:
        return self.reranked[0][1] if self.reranked else None


class Retriever:
    def __init__(self, index: Index, reranker: Reranker, gloss: dict | None = None, top_bm25=conf.TOP_BM25,
                 top_dense=conf.TOP_DENSE, rrf_k=conf.RRF_K, pool=conf.RERANK_POOL):
        self.ix, self.rr, self.gloss = index, reranker, gloss or {}
        self.top_bm25, self.top_dense, self.rrf_k, self.pool = top_bm25, top_dense, rrf_k, pool
        self._dense = index.vector.as_retriever(similarity_top_k=top_dense)

    def _bm25(self, q: str) -> list[str]:
        import numpy as np
        toks = vi.tokens(q)
        if not toks: return []
        sc = np.asarray(self.ix.bm25.get_scores(toks))
        order = np.argsort(-sc, kind="stable")[: self.top_bm25]
        return [self.ix.ids[i] for i in order if sc[i] > 0]

    def search(self, query: str) -> Result:
        q = expand(query, self.gloss)
        r = Result(q)
        if not vi.tokens(q):                                # nothing searchable (punctuation only)
            return r
        t = time.time(); r.bm25 = self._bm25(q); r.seconds["bm25"] = round(time.time() - t, 3)
        t = time.time(); r.dense = [h.node.node_id for h in self._dense.retrieve(q)]; r.seconds["dense"] = round(time.time() - t, 3)
        fused = rrf([r.bm25, r.dense], self.rrf_k)
        r.fused = [c for c, _ in fused]
        pool = r.fused[: self.pool]
        t = time.time()
        if self.rr is None:                                 # no cross-encoder: keep the fused order, RRF score as the score
            r.reranked = [(c, s) for c, s in fused[: self.pool]]
        else:
            sc = self.rr.score(q, [self.ix.nodes[c]["text"] for c in pool]) if pool else []
            r.reranked = sorted(zip(pool, sc), key=lambda x: -x[1])
        r.seconds["rerank"] = round(time.time() - t, 3)
        return r

    def warm(self) -> None:
        """Load the models before the first question, so the first answer is not slower than the rest."""
        self._dense.retrieve("tổng tài sản")
        if self.rr is not None:
            self.rr.score("tổng tài sản", ["Tổng tài sản"])

    def top_nodes(self, r: Result, n: int = conf.FINAL_EVIDENCE) -> list[dict]:
        return [self.ix.nodes[c] for c, _ in r.reranked[:n]]

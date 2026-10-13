"""Hybrid retrieval: BM25 (tcb-vi-v2 tokens) + dense BGE-M3 (LlamaIndex) -> RRF -> top FINAL_EVIDENCE blocks.
No cross-encoder: it bought one item of recall in 85 for most of the CPU time per question (DESIGN 17.1).
Every stage's ranking is kept so eval can attribute a miss to the stage that lost the gold page."""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from . import conf, vi
from .index import Index


_ABBR = re.compile(r"^[A-Za-zĐ0-9/ ]{2,10}$")


def expand(query: str, gloss: dict) -> str:
    """Append the report's own glossary definition after an abbreviation: 'CASA' -> 'CASA (Tiền gửi không kỳ hạn)'."""
    q = vi.nfc(query)
    extra = []
    for term, v in gloss.items():
        if _ABBR.match(term) and any(c.isupper() for c in term) and re.search(rf"(?<!\w){re.escape(term)}(?!\w)", q):
            extra.append(v["definition"][:100].strip())
    return q if not extra else f"{q} ({'; '.join(extra)})"


def _norm(s: str) -> str:
    return " " + re.sub(r"[^\w%]+", " ", vi.fold(vi.nfc(s).lower())).strip() + " "


_OPEN = {"tu", "tren", "va", "cua", "cho", "voi", "trong", "theo", "den", "la", "da"}


def kpi_labels(nodes: dict) -> dict[str, str]:
    """chunk_id -> label of a KPI tile ('Chi phí tín dụng' in '[…] Chi phí tín dụng | 0,6%'). A tile is two or three
    words and a figure, so it ranks below longer blocks that repeat the same words; naming its label in the question is
    the strongest signal there is that the tile is the answer."""
    out = {}
    for cid, n in nodes.items():
        if n.get("content_type") != "kpi": continue
        body = re.sub(r"^\[[^\]]*\]\s*", "", n["text"])
        lab = _norm(body.split(" | ")[0])
        w = lab.split()
        # a metric name: 3+ syllables, no figure, not a sentence opening ('Techcombank đã'), not cut mid-phrase ('… từ')
        if len(w) >= 3 and not re.search(r"\d", lab) and w[0] != vi.fold(conf.DOC_SUBJECT.lower()) and w[-1] not in _OPEN:
            out[cid] = lab
    return out


def label_matches(query: str, labels: dict[str, str]) -> list[str]:
    """Tiles whose whole label occurs in the question, longest label first: a third ranking for RRF."""
    q = _norm(query)
    hits = [(len(lab), cid) for cid, lab in labels.items() if lab in q]
    return [cid for _, cid in sorted(hits, key=lambda x: (-x[0], x[1]))]


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
    label: list[str] = field(default_factory=list)              # KPI tiles whose label is named in the question
    fused: list[str] = field(default_factory=list)
    ranked: list[tuple[str, float]] = field(default_factory=list)     # (chunk_id, RRF score) given to the LLM, best first
    seconds: dict = field(default_factory=dict)


class Retriever:
    def __init__(self, index: Index, gloss: dict | None = None, top_bm25=conf.TOP_BM25,
                 top_dense=conf.TOP_DENSE, rrf_k=conf.RRF_K, k=conf.FINAL_EVIDENCE):
        self.ix, self.gloss = index, gloss or {}
        self.top_bm25, self.top_dense, self.rrf_k, self.k = top_bm25, top_dense, rrf_k, k
        self._dense = index.vector.as_retriever(similarity_top_k=top_dense)
        self.labels = kpi_labels(index.nodes)

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
        from llama_index.core import QueryBundle
        t = time.time(); emb = self.ix.vector._embed_model.get_query_embedding(q); r.seconds["embed"] = round(time.time() - t, 3)
        t = time.time(); r.dense = [h.node.node_id for h in self._dense.retrieve(QueryBundle(q, embedding=emb))]
        r.seconds["dense"] = round(time.time() - t, 3)
        r.label = label_matches(q, self.labels)
        fused = rrf([r.bm25, r.dense] + ([r.label] if r.label else []), self.rrf_k)
        r.fused = [c for c, _ in fused]
        r.ranked = fused[: self.k]
        return r

    def warm(self) -> None:
        """Load the embedding model before the first question, so the first answer is not slower than the rest."""
        self._dense.retrieve("tổng tài sản")

    def top_nodes(self, r: Result) -> list[dict]:
        return [self.ix.nodes[c] for c, _ in r.ranked]

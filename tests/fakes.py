"""Test doubles. EVERYTHING here is a stand-in: none of it proves a real model/provider/retriever works."""
import json
import re
from dataclasses import asdict

from tcb import llm, find


class Scripted:
    """LLM stand-in: `handler(req, n)` returns the reply text (n = 0-based call number)."""
    name, model = "scripted", "scripted-1"

    def __init__(self, handler):
        self.handler, self.calls = handler, []

    def generate(self, req):
        self.calls.append(req)
        out = self.handler(req, len(self.calls) - 1)
        if isinstance(out, Exception): raise out
        return llm.LLMResponse(out if isinstance(out, str) else json.dumps(out, ensure_ascii=False), self.name, self.model, 0.01, 100, 20)


def eid_of(req, needle: str) -> str:
    """Evidence id whose text contains `needle` (parsed from the prompt the engine built)."""
    for m in re.finditer(r"\[(E\d+)\]\n(.*?)(?=\n\n\[E\d+\]|\n\nĐẠI LƯỢNG)", req.user, re.S):
        if needle in m.group(2): return m.group(1)
    raise AssertionError(f"{needle!r} not in evidence")


def qid_of(req, label: str, period: str) -> str:
    for m in re.finditer(r"\[(Q\d+)\] (.*?) \| (.*?) \|", req.user):
        if label in m.group(2) and period in m.group(3): return m.group(1)
    raise AssertionError(f"quantity {label!r}/{period!r} not offered")


class NodeSet:
    """Retriever stand-in returning fixed nodes (so pipeline tests do not depend on ranking quality)."""
    pool = 30

    def __init__(self, nodes):
        self.nodes = [asdict(n) if not isinstance(n, dict) else n for n in nodes]

    def search(self, q):
        r = find.Result(q)
        ids = [n["chunk_id"] for n in self.nodes]
        r.bm25 = r.dense = r.fused = ids
        r.reranked = [(c, 1.0 - i * 0.01) for i, c in enumerate(ids)]
        return r

    def top_nodes(self, r, n=8):
        return self.nodes[:n]

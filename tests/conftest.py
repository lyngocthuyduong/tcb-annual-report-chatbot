import hashlib
import math
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PDF = ROOT / "data" / "ar.pdf"


@pytest.fixture(scope="session")
def doc():
    if not PDF.exists():
        pytest.skip("data/ar.pdf missing")
    from tcb import nodes, pdf
    halves, info = pdf.load(str(PDF))
    ns, gloss = nodes.build_nodes(halves)
    return {"halves": halves, "info": info, "nodes": ns, "gloss": gloss,
            "H": {(h.pdf_page_index, h.side): h for h in halves}}


def text_of(h) -> str:
    return " ".join(b.text for b in h.blocks)


class HashEmbed:  # placeholder replaced below by a LlamaIndex BaseEmbedding subclass
    pass


def make_embedder(dim: int = 64):
    from llama_index.core.base.embeddings.base import BaseEmbedding

    class _E(BaseEmbedding):
        model_name: str = "TEST-HASH"

        def _vec(self, text: str):
            v = [0.0] * dim
            for t in re.findall(r"\w+", text.lower()):
                v[int(hashlib.md5(t.encode()).hexdigest(), 16) % dim] += 1.0
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            return [x / n for x in v]

        def _get_query_embedding(self, query): return self._vec(query)
        async def _aget_query_embedding(self, query): return self._vec(query)
        def _get_text_embedding(self, text): return self._vec(text)

    return _E()


@pytest.fixture(scope="session")
def embedder():
    return make_embedder()


@pytest.fixture(scope="session")
def built(tmp_path_factory, doc, embedder):
    from tcb import build
    out = tmp_path_factory.mktemp("idx") / "idx"
    m = build.build(str(PDF), str(out), embedder=embedder, embed_name="TEST-HASH", embed_rev="test")
    return out, m


def pytest_collection_modifyitems(items):
    """Default category by file when a test file has not marked itself."""
    unit_files = {"test_vi.py", "test_conf.py", "test_calc.py", "test_cite.py", "test_llm.py", "test_eval.py", "test_api.py"}
    for it in items:
        if any(m.name in ("unit", "integ", "live", "live_openai", "live_ollama", "live_models") for m in it.iter_markers()):
            continue
        it.add_marker(pytest.mark.unit if it.path.name in unit_files else pytest.mark.integ)


@pytest.fixture(scope="session")
def tindex(built, embedder):
    """Index loaded from the TEST-HASH index in tmp. Never the submission index."""
    from tcb.index import Index
    out, _ = built
    return Index.load(out, embedder, embed_model="TEST-HASH", embed_rev="test")

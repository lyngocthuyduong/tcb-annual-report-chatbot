"""INTEGRATION on a TEST-HASH index built in tmp from the real PDF. The hash embedder and overlap reranker are
stand-ins: these tests check retrieval MECHANICS (lexical number matching, RRF, stage bookkeeping), not retrieval quality."""
import json

import pytest

from tcb import find, vi


class OverlapRerank:
    def score(self, q, ts):
        qs = set(vi.tokens(q))
        return [len(qs & set(vi.tokens(t))) / (1 + len(t.split()) ** 0.5) for t in ts]


@pytest.fixture(scope="module")
def rt(tindex, built):
    gloss = json.loads((built[0] / "gloss.json").read_text(encoding="utf-8"))
    return find.Retriever(tindex, OverlapRerank(), gloss)


def page_of(rt, cid): return rt.ix.nodes[cid]["printed_page_number"]


def test_rrf_math():
    out = dict(find.rrf([["a", "b"], ["b", "c"]], 60))
    assert out["b"] == pytest.approx(1 / 62 + 1 / 61) and out["b"] > out["a"] > out["c"]
    assert find.rrf([], 60) == [] and [c for c, _ in find.rrf([["x", "y"]])] == ["x", "y"]


def test_lexical_side_finds_exact_vietnamese_number(rt):
    r = rt.search("1.192.344.137")
    assert 297 in {page_of(rt, c) for c in r.bm25[:5]}               # number kept whole by tcb-vi-v1
    r = rt.search("1192344137")                                       # canonical form also matches
    assert 297 in {page_of(rt, c) for c in r.bm25[:5]}


def test_glossary_expansion(rt):
    r = rt.search("Tỷ lệ CASA là bao nhiêu?")
    assert "Tiền gửi không kỳ hạn" in r.query
    assert find.expand("không có từ viết tắt nào", rt.gloss) == "không có từ viết tắt nào"
    assert find.expand("AI là gì", rt.gloss) != "AI là gì"            # whole-word, case-sensitive
    assert find.expand("mail", rt.gloss) == "mail"


def test_stage_bookkeeping(rt):
    r = rt.search("tổng tài sản 2025")
    assert len(r.bm25) <= 40 and len(r.dense) == 40 and len(r.fused) >= len(r.bm25)
    assert len(r.reranked) == min(30, len(r.fused)) and set(c for c, _ in r.reranked) <= set(r.fused)
    s = [x for _, x in r.reranked]; assert s == sorted(s, reverse=True)
    assert set(r.seconds) == {"bm25", "dense", "rerank"}
    assert len(rt.top_nodes(r)) == 8 and r.top_score == s[0]


def test_empty_query_does_not_crash(rt):
    r = rt.search("???")
    assert r.bm25 == [] and r.dense == [] and r.reranked == []


def test_missing_reranker_snapshot_fails_loudly(tmp_path):
    with pytest.raises(FileNotFoundError, match="baked into the Docker image"):
        find.CrossEncoderReranker(tmp_path / "nope").score("q", ["t"])


def test_test_index_is_never_the_submission_index(built):
    out, m = built
    assert m["embedding"]["model"] == "TEST-HASH"
    from tcb.index import IndexError_, validate
    with pytest.raises(IndexError_) as e: validate(out)                 # production validate must refuse it
    assert e.value.code == "embedding"

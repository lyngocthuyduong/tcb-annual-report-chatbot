"""Integration on a TEST-HASH index built in tmp from the real PDF. The hash embedder is a stand-in: these tests check
retrieval MECHANICS (lexical number matching, RRF, stage bookkeeping), not retrieval quality."""
import json

import pytest

from tcb import conf, find


@pytest.fixture(scope="module")
def rt(tindex, built):
    gloss = json.loads((built[0] / "gloss.json").read_text(encoding="utf-8"))
    return find.Retriever(tindex, gloss)


def page_of(rt, cid): return rt.ix.nodes[cid]["printed_page_number"]


def test_rrf_math():
    out = dict(find.rrf([["a", "b"], ["b", "c"]], 60))
    assert out["b"] == pytest.approx(1 / 62 + 1 / 61) and out["b"] > out["a"] > out["c"]
    assert find.rrf([], 60) == [] and [c for c, _ in find.rrf([["x", "y"]])] == ["x", "y"]


def test_lexical_side_finds_exact_vietnamese_number(rt):
    r = rt.search("1.192.344.137")
    assert 297 in {page_of(rt, c) for c in r.bm25[:5]}               # number kept whole by the tcb-vi tokenizer
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
    assert [c for c, _ in r.ranked] == r.fused[: conf.FINAL_EVIDENCE]                 # no reranker: the fused order is final
    s = [x for _, x in r.ranked]; assert s == sorted(s, reverse=True)
    assert set(r.seconds) == {"bm25", "embed", "dense"}
    assert len(rt.top_nodes(r)) == min(conf.FINAL_EVIDENCE, len(r.fused))
    rt.warm()


def test_empty_query_does_not_crash(rt):
    r = rt.search("???")
    assert r.bm25 == [] and r.dense == [] and r.ranked == []


def test_test_index_is_never_the_submission_index(built):
    out, m = built
    assert m["embedding"]["model"] == "TEST-HASH"
    from tcb.index import IndexError_, validate
    with pytest.raises(IndexError_) as e: validate(out)                 # production validate must refuse it
    assert e.value.code == "embedding"


def test_kpi_label_named_in_the_question_is_a_ranking():
    nodes = {"k1": {"content_type": "kpi", "text": "[Thành tựu › Báo cáo Kết quả kinh doanh] Chi phí tín dụng | 0,6%"},
             "k2": {"content_type": "kpi", "text": "[x] Tỷ lệ CASA | 40,4%"},
             "k3": {"content_type": "kpi", "text": "[x] 18,0 | Triệu khách hàng"},
             "n1": {"content_type": "narrative", "text": "[x] Chi phí dự phòng rủi ro tín dụng tăng"}}
    labels = find.kpi_labels(nodes)
    assert set(labels) == {"k1", "k2"}                                   # a label needs words, not a figure
    assert find.label_matches("Chi phí tín dụng của Techcombank năm 2025 là bao nhiêu?", labels) == ["k1"]
    assert find.label_matches("chi phi tin dung nam 2025?", labels) == ["k1"]          # diacritics folded
    assert find.label_matches("Chi phí dự phòng rủi ro tín dụng?", labels) == []

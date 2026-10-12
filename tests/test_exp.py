"""Experiment runner. Unit parts check the math; integration parts run on the TEST-HASH index from the real PDF.
The numbers produced here say NOTHING about real retrieval quality -- only that the measuring code works."""
import json

import pytest

from tcb import exp
from tcb.index import IndexError_, validate


# ---------------------------------------------------------------- unit
@pytest.mark.unit
def test_rank_and_agg():
    page = {"a": 1, "b": 2, "c": 3, "d": 3}.get
    assert exp.first_rank(["a", "b", "c"], {3}, page) == 3 and exp.first_rank(["a"], {9}, page) is None
    m = exp.agg([1, 3, None, 9, 41], ks=(5, 8))
    assert m == {"n": 5, "mrr": round((1 + 1 / 3 + 1 / 9) / 5, 4), "R@5": 0.4, "R@8": 0.4}
    assert exp.agg([]) == {"n": 0}


@pytest.mark.unit
def test_fusions():
    bm = [("a", 9.0), ("b", 5.0), ("c", 1.0)]; dn = [("c", .9), ("b", .8), ("d", .1)]
    assert exp.fuse_rrf([bm, dn], 60) == ["c", "b", "a", "d"]        # c: 1/63+1/61 > b: 2/62 > a: 1/61 > d: 1/63
    r = exp.fuse_rsf([bm, dn])
    assert r == ["b", "a", "c", "d"]            # b=.5+.875, a=1+0, c=0+1 (tie with a -> id order), d=0
    assert exp.fuse_rsf([[("x", 1.0)], []]) == ["x"]                          # degenerate list: single score -> 1.0
    assert exp.fuse_rrf([[], []], 60) == []


@pytest.mark.unit
def test_tokenizer_variants_show_why_vi_exists():
    assert "1.192" in exp.tok_vi("tổng tài sản 1.192 nghìn tỷ") and "1192" in exp.tok_vi("1.192")
    d = exp.tok_default("tổng tài sản 1.192 nghìn tỷ")
    assert "1.192" not in d and "192" in d                                     # bm25s default splits Vietnamese numbers
    b = exp.tok_bigram("tỷ lệ nợ xấu 1,13%")
    assert "tỷ_lệ" in b and "nợ_xấu" in b and "tỷ_ty" not in b and not any("1,13" in t for t in b if "_" in t)


# ---------------------------------------------------------------- integration (hash test index)
@pytest.fixture(scope="module")
def ctx(tindex, built):
    gloss = json.loads((built[0] / "gloss.json").read_text(encoding="utf-8"))
    return exp.make_ctx(tindex, gloss)


def dataset(tmp_path):
    items = [
        {"id": "a1", "question": "Tổng tài sản tại 31/12/2025 là bao nhiêu triệu đồng 1.192.344.137", "gold_printed_pages": [297], "verified": True},
        {"id": "a2", "question": "Tiền mặt, vàng 4.360.821 triệu đồng", "gold_printed_pages": [297], "verified": True},
        {"id": "a3", "question": "CASA là viết tắt của thuật ngữ gì?", "gold_printed_pages": [386], "verified": True},
        {"id": "a4", "question": "Tỷ lệ CASA 40,4%", "gold_printed_pages": [5], "alt_pages": [48], "verified": True},
        {"id": "u1", "question": "Giá cổ phiếu hôm nay", "answerable": False, "verified": True},
        {"id": "u2", "question": "Dự báo lợi nhuận năm 2027", "answerable": False, "verified": True},
        {"id": "u3", "question": "Thời tiết Hà Nội", "answerable": False, "verified": True},
        {"id": "x1", "question": "chưa xác minh", "gold_printed_pages": [5], "verified": False},
    ]
    p = tmp_path / "dev.json"; p.write_text(json.dumps(items), encoding="utf-8")
    q = tmp_path / "val.json"; q.write_text(json.dumps(items[:4] + items[4:7]), encoding="utf-8")
    return str(p), str(q)


@pytest.mark.integ
def test_full_run_structure_and_outputs(ctx, tmp_path):
    dev, val = dataset(tmp_path)
    res = exp.run_exp([dev, val], ctx)
    assert res["meta"]["n_answerable"] == 8 and res["meta"]["n_unanswerable"] == 6          # unverified x1 excluded
    for key in ("E2", "E3", "E3b", "E4", "E5"): assert key in res, key
    assert set(res["E2"]) == {"default", "vi", "vi+bigram"} and set(res["E3"]) == {"bm25", "dense", "rrf60", "rsf"}
    assert set(res["E3b"]) == {"10", "30", "60", "100"} and set(res["E4"]["all"]) >= {f"R@{k}" for k in exp.FINAL_KS}
    assert set(res["E2"]["vi"]) == {"dev", "val", "all"}
    # the lexical side must find the exact-number question's page (tokenizer keeps 1.192.344.137 whole)
    assert res["E2"]["vi"]["all"]["R@20"] > 0
    dec = res["decisions"]
    assert dec["E3b"]["keep_k"] in (10, 30, 60, 100) and dec["E4"]["choose_k"] in exp.FINAL_KS and "keep_expansion" in dec["E5"]
    assert "E7" not in res
    d = exp.write(res, str(tmp_path / "out"), "t")
    assert json.loads((d / "exp.json").read_text(encoding="utf-8"))["meta"]["n_answerable"] == 8
    md = (d / "exp.md").read_text(encoding="utf-8")
    assert "E3b RRF k" in md and "Suggested decisions (not applied)" in md


@pytest.mark.integ
def test_alt_pages_and_subset_selection(ctx, tmp_path):
    dev, _ = dataset(tmp_path)
    res = exp.run_exp([dev], ctx, which=("E2", "E3"))
    assert "E4" not in res and res["meta"]["n_answerable"] == 4


@pytest.mark.integ
def test_requires_verified_items(ctx, tmp_path):
    p = tmp_path / "s.json"; p.write_text(json.dumps([{"id": "z", "question": "q", "gold_printed_pages": [5]}]), encoding="utf-8")
    with pytest.raises(ValueError, match="no verified"): exp.run_exp([str(p)], ctx)
    assert exp.run_exp([str(p)], ctx, which=("E3",), include_unverified=True)["meta"]["include_unverified"] is True


@pytest.mark.integ
def test_chunk_variant_index_is_rejected_by_production_but_loadable_for_experiments(doc, embedder, tmp_path):
    from tcb import build
    from tcb.index import Index
    out = tmp_path / "i"
    m = build.build("data/ar.pdf", str(out), embedder=embedder, embed_name="TEST-HASH", embed_rev="test", chunk_words=90)
    assert m["chunking"]["narrative_max_words"] == 90 and m["counts"]["nodes"] > 1542
    with pytest.raises(IndexError_) as e: validate(out, embed_model="TEST-HASH", embed_rev="test")
    assert e.value.code == "format" and "chunking" in str(e.value)
    validate(out, embed_model="TEST-HASH", embed_rev="test", strict_chunking=False)
    ix = Index.load(out, embedder, embed_model="TEST-HASH", embed_rev="test", strict_chunking=False)
    assert len(ix.nodes) == m["counts"]["nodes"]

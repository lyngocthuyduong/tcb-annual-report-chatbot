"""UNIT: eval schema, scorers, attribution and runner with scripted LLM + synthetic nodes.
Verifies the MEASURING code. The numbers it prints in these tests mean nothing about system quality."""
import json
from types import SimpleNamespace as NS

import pytest

from fakes import NodeSet, Scripted, eid_of
from tcb import ask, conf, ev

pytestmark = pytest.mark.unit


def item(**kw):
    base = dict(id="x", question="q?", gold_answer="40,4%", gold_printed_pages=[5], gold_facts=["40,4"], verified=True)
    base.update(kw); return ev.Item(**base)


def trace(bm=("a",), de=("a",), fu=("a",), rr=(("a", 1.0),), ev_pages=(5,)):
    return {"retrieval": {"bm25": list(bm), "dense": list(de), "fused": list(fu), "ranked": [list(x) for x in rr]},
            "evidence": [{"page": p} for p in ev_pages], "total_s": 1.0, "llm": {"calls": 1, "cost_usd": 0.001}}


PAGES = {"a": 5, "b": 9, "c": 11}
page_of = PAGES.get


def A(status="answered", answer="CASA 40,4% [tr. 5]", tr=None, **kw):
    return {"status": status, "answer": answer, "trace": tr or trace(), **kw}


def test_schema_rules(tmp_path):
    p = tmp_path / "s.json"
    p.write_text(json.dumps([{"id": "a", "question": "q", "answerable": False, "gold_printed_pages": [3]}]))
    with pytest.raises(ValueError, match="unanswerable"): ev.load_items(p)
    p.write_text(json.dumps([{"id": "a", "question": "q"}, {"id": "a", "question": "r"}]))
    with pytest.raises(ValueError, match="duplicate"): ev.load_items(p)
    p.write_text(json.dumps([{"id": "a", "question": "q", "type": "weird"}]))
    with pytest.raises(Exception): ev.load_items(p)
    assert not ev.Item(id="a", question="q").verified              # default: NOT verified


def test_shipped_datasets_load():
    qs = ev.load_items("data/q10.json"); assert len(qs) == 10 and all(q.verified for q in qs)
    assert ev.load_items("data/dev.json") and ev.load_items("data/val.json")      # drafts until human-verified (tcb vset apply)


def test_facts_alternatives_and_units():
    it = item(gold_facts=["1.192 | 1.192.344.137", "21,8"])
    assert ev.facts_ok(it, "Tổng tài sản 1.192.344.137 triệu đồng, tăng 21,8%")[0] is True
    assert ev.facts_ok(it, "Tổng tài sản 1.192 nghìn tỷ, tăng 21,8%")[0] is True
    assert ev.facts_ok(it, "Tổng tài sản 1.192 nghìn tỷ")[0] is False
    assert ev.facts_ok(item(gold_facts=["Tiền gửi không kỳ hạn"]), "CASA là tiền gửi KHÔNG kỳ hạn.")[0] is True
    assert ev.facts_ok(item(gold_facts=[], gold_answer="40,4%"), "Tỷ lệ 40,4%")[0] is True
    assert ev.facts_ok(item(gold_facts=[], gold_answer="Khối Bán lẻ"), "x") == (None, "unscorable")


@pytest.mark.parametrize("kw,ans,verdict", [
    ({}, A(), "pass"),
    ({}, A(tr=trace(bm=(), de=(), fu=(), rr=())), "retrieval:recall_miss"),
    ({}, A("not_in_report", "Không có", tr=trace(bm=("b",), de=("b",), fu=("b",), rr=(("b", 1),), ev_pages=(9,))), "retrieval:recall_miss"),
    ({}, A(tr=trace(fu=("b",), rr=(("b", 1),))), "retrieval:fusion_drop"),
    ({}, A(tr=trace(rr=(("b", 1),) * conf.FINAL_EVIDENCE + (("a", 0.1),))), "retrieval:rank_drop"),
    ({}, A("not_in_report", "Không có thông tin"), "refusal:over_refusal"),
    ({}, A(answer="CASA 39% [tr. 5]"), "generation:wrong_or_missing_fact"),
    ({}, A(answer="CASA 40,4% [tr. 9]", tr=trace(ev_pages=(5, 9))), "citation:wrong_page"),
    ({}, A(answer="CASA 40,4% [tr. 77]"), "citation:invalid_page"),
    ({}, A("error", "x", error_kind="timeout"), "system:timeout"),
    ({"answerable": False, "gold_printed_pages": [], "gold_facts": []}, A("not_in_report", "Không có"), "pass"),
    ({"answerable": False, "gold_printed_pages": [], "gold_facts": []}, A("answered", "Dự báo 100 tỷ"), "refusal:under_refusal"),
])
def test_attribution(kw, ans, verdict):
    assert ev.score_item(item(**kw), ans, page_of)["verdict"] == verdict


def test_calc_error_attribution():
    a = A("answered", "x", tr={**trace(), "calc_errors": ["C1: chia 0"]}); a["trace"]["calc_errors"] = ["C1"]
    assert ev.score_item(item(gold_facts=[]), a, page_of)["verdict"] == "calc:error"


def test_alt_pages_count_as_gold():
    r = ev.score_item(item(alt_pages=[9]), A(answer="CASA 40,4% [tr. 9]", tr=trace(bm=("b",), de=("b",), fu=("b",), rr=(("b", 1),), ev_pages=(9,))), page_of)
    assert r["passed"] and r["page_hit"]


def test_summary_metrics():
    rows = [ev.score_item(item(id="1"), A(), page_of),
            ev.score_item(item(id="2"), A("not_in_report", "Không có"), page_of),
            ev.score_item(item(id="3", answerable=False, gold_printed_pages=[], gold_facts=[]), A("not_in_report", "Không có"), page_of),
            ev.score_item(item(id="4", answerable=False, gold_printed_pages=[], gold_facts=[]), A("answered", "bịa"), page_of)]
    s = ev.summarize(rows)
    assert s["n"] == 4 and s["pass_rate"] == 0.5
    assert s["refusal"] == {"precision": 0.5, "recall": 0.5, "n_unanswerable": 2, "false_answer_rate": 0.5}
    assert s["retrieval"][f"recall@{conf.FINAL_EVIDENCE}_final"] == 1.0 and s["retrieval"]["mrr_final"] == 1.0
    assert s["attribution"]["pass"] == 2 and s["answers"]["citation_valid"] == 1.0


NODES = {"n1": {"chunk_id": "n1", "text": "Tỷ lệ CASA | 40,4%", "printed_page_number": 5, "pdf_page_index": 3, "side": "R",
                "section": "S", "content_type": "kpi", "page_number_source": "footer"}}


class RT(NodeSet):
    ix = NS(manifest={"counts": {"nodes": 1}, "embedding": {"model": "TEST-HASH", "revision": "test"}, "built_at": "t"}, nodes=NODES)
    rr, rrf_k, top_bm25, top_dense = object(), 60, 40, 40


def test_run_eval_end_to_end_writes_machine_readable_artifacts(tmp_path):
    sp = tmp_path / "set.json"
    sp.write_text(json.dumps([
        {"id": "v1", "question": "CASA bao nhiêu?", "gold_printed_pages": [5], "gold_facts": ["40,4"], "verified": True, "category": "kf"},
        {"id": "u1", "question": "ROE bao nhiêu?", "gold_printed_pages": [5], "gold_facts": ["99"], "verified": False},
        {"id": "r1", "question": "Dự báo 2027?", "answerable": False, "type": "refusal", "verified": True}]))
    def h(rq, i):
        if "Dự báo" in rq.user: return {"status": "not_in_report", "claims": []}
        return {"status": "answered", "claims": [{"text_vi": "CASA là 40,4%.", "evidence": [eid_of(rq, "40,4%")]}]}
    e = ask.Engine(RT([NODES["n1"]]), Scripted(h))
    rep = ev.run_eval(str(sp), str(tmp_path / "runs"), e, label="t")
    s = rep["summary"]
    assert s["n"] == 2 and s["items_run"] == 3 and s["items_unverified_excluded"] == 1 and s["pass_rate"] == 1.0
    assert s["submission_grade"] is False and rep["config"]["index_embedding"]["model"] == "TEST-HASH"
    d = tmp_path / "runs" / "t"
    assert {p.name for p in d.iterdir()} == {"traces.jsonl", "results.jsonl", "summary.json", "config.json"}
    tr = [json.loads(l) for l in (d / "traces.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(tr) == 3 and tr[0]["answer"]["trace"]["retrieval"]["ranked"] and tr[0]["score"]["verdict"] == "pass"
    cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
    assert cfg["provider"] == "scripted" and len(cfg["set_sha256"]) == 64
    rep2 = ev.run_eval(str(sp), str(tmp_path / "runs"), e, label="t2", include_unverified=True)
    assert rep2["summary"]["n"] == 3 and rep2["summary"]["attribution"].get("generation:wrong_or_missing_fact") == 1


def _fake_run(d, rows, cfg_extra=None, summary=None):
    d.mkdir(parents=True)
    (d / "results.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    (d / "summary.json").write_text(json.dumps(summary or {"pass_rate": 0.5, "retrieval": {f"recall@{conf.FINAL_EVIDENCE}_final": 0.8, "mrr_final": 0.6}}), encoding="utf-8")
    (d / "config.json").write_text(json.dumps({"set_sha256": "s1", "provider": "p", "model": "m", **(cfg_extra or {})}), encoding="utf-8")


def test_compare_runs(tmp_path):
    a = tmp_path / "a"; b = tmp_path / "b"
    R = lambda i, ok, v: {"id": i, "passed": ok, "verdict": v}
    _fake_run(a, [R("1", True, "pass"), R("2", False, "retrieval:recall_miss"), R("3", False, "generation:wrong_or_missing_fact"), R("4", True, "pass")])
    _fake_run(b, [R("1", False, "citation:wrong_page"), R("2", True, "pass"), R("3", False, "retrieval:rank_drop"), R("5", True, "pass")],
              {"model": "m2", "set_sha256": "s2"}, {"pass_rate": 0.75, "retrieval": {f"recall@{conf.FINAL_EVIDENCE}_final": 0.9, "mrr_final": 0.7}})
    c = ev.compare(str(a), str(b))
    assert c["fixed"] == ["2"] and c["regressed"] == ["1"] and c["only_in_a"] == ["4"] and c["only_in_b"] == ["5"]
    assert c["verdict_changed"] == [{"id": "3", "from": "generation:wrong_or_missing_fact", "to": "retrieval:rank_drop"}]
    assert c["metrics"]["pass_rate"] == {"a": 0.5, "b": 0.75, "delta": 0.25}
    assert any("DIFFERENT DATASET" in w for w in c["config_differences"]) and any("model" in w for w in c["config_differences"])
    t = ev.compare_text(c); assert "REGRESSED (1): 1" in t and "fixed (1): 2" in t


def test_cli_compare_exit_code_flags_regressions(tmp_path, capsys):
    from tcb import cli
    a, b = tmp_path / "a", tmp_path / "b"
    _fake_run(a, [{"id": "1", "passed": True, "verdict": "pass"}]); _fake_run(b, [{"id": "1", "passed": False, "verdict": "x"}])
    assert cli.main(["eval", "--compare", str(a), str(b), "--out", str(tmp_path / "o")]) == 1
    assert "REGRESSED (1)" in capsys.readouterr().out and list((tmp_path / "o").glob("compare_*.json"))
    _fake_run(tmp_path / "c", [{"id": "1", "passed": True, "verdict": "pass"}])
    assert cli.main(["eval", "--compare", str(a), str(tmp_path / "c"), "--out", str(tmp_path / "o")]) == 0
    assert cli.main(["eval"]) == 2

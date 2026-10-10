"""UNIT: citation building/validation and refusal policy on synthetic nodes."""
import pytest

from tcb import cite, plan as P, refuse
from tcb.num import Period, Quantity
from decimal import Decimal as D

pytestmark = pytest.mark.unit


def node(cid, text, page=10, ct="narrative", pdf=5):
    return {"chunk_id": cid, "text": text, "printed_page_number": page, "pdf_page_index": pdf, "side": "L",
            "section": "S", "content_type": ct, "page_number_source": "footer"}


EVS = cite.evidence_from([node("a", "CASA đạt 40,4% năm 2025", 5), node("b", "Dư nợ vay 328,1 nghìn tỷ đồng, tăng 26,9% N/N", 59),
                          node("c", "Trang bìa", None)])


def plan(**kw):
    base = {"status": "answered", "claims": [{"text_vi": "CASA là 40,4%.", "evidence": ["E1"]}]}
    base.update(kw); return P.Plan.model_validate(base)


def test_evidence_ids_and_pages_from_metadata():
    assert [(e.eid, e.page) for e in EVS] == [("E1", 5), ("E2", 59), ("E3", None)]


def test_clean_plan_ok():
    assert cite.check(plan(), EVS, {}, "CASA?").ok


@pytest.mark.parametrize("claim,frag", [
    ({"text_vi": "CASA 40,5%.", "evidence": ["E1"]}, "numbers not found"),
    ({"text_vi": "CASA 40,4%.", "evidence": ["E9"]}, "unknown evidence id E9"),
    ({"text_vi": "CASA 40,4%.", "evidence": []}, "no evidence cited"),
    ({"text_vi": "CASA 40,4% [tr. 5].", "evidence": ["E1"]}, "page reference"),
    ({"text_vi": "CASA 40,4% ở trang 5.", "evidence": ["E1"]}, "page reference"),
    ({"text_vi": "Trang bìa.", "evidence": ["E3"]}, "no printed page"),
    ({"text_vi": "Tăng {C1}.", "evidence": ["E1"]}, "unknown calculation C1"),
    ({"text_vi": "Dư nợ 328,1 nghìn tỷ.", "evidence": ["E1"]}, "numbers not found"),    # number is in E2, not the cited E1
])
def test_claim_problems(claim, frag):
    ck = cite.check(plan(claims=[claim]), EVS, {}, "")
    assert not ck.ok and any(frag in p for p in ck.flat), ck.flat


def test_number_formats_equivalent_and_trivial_ints_ignored():
    ok = cite.check(plan(claims=[{"text_vi": "Có 3 chỉ tiêu: CASA 40,4 phần trăm.", "evidence": ["E1"]}]), EVS, {}, "")
    assert ok.ok
    assert cite.check(plan(claims=[{"text_vi": "Dư nợ 328,1 nghìn tỷ đồng.", "evidence": ["E2"]}]), EVS, {}, "").ok


def test_numbers_from_question_are_allowed_years_not():
    assert cite.check(plan(claims=[{"text_vi": "Năm 2025, CASA là 40,4%.", "evidence": ["E1"]}]), EVS, {}, "").ok
    assert not cite.check(plan(claims=[{"text_vi": "Năm 2019, CASA là 40,4%.", "evidence": ["E1"]}]), EVS, {}, "").ok


def test_calc_inputs_must_exist_and_placeholder_numbers_allowed():
    q = Quantity("Q1", "x", D(5), "VND", 1, Period("date", 2025, "2025"), "a", 5, "5")
    pl = plan(claims=[{"text_vi": "Tăng {C1} so với năm trước.", "evidence": [], "calc_refs": ["C1"]}],
              calculations=[{"cid": "C1", "op": "growth_pct", "inputs": ["Q1", "Q9"]}])
    ck = cite.check(pl, EVS, {"Q1": q}, "")
    assert any("unknown quantity Q9" in p for p in ck.flat)


def test_pages_and_marker_come_from_nodes_only():
    pl = plan(claims=[{"text_vi": "a 40,4% b 328,1", "evidence": ["E2", "E1"]}])
    pg = cite.pages_of(pl.claims[0], EVS, pl, {}, {})
    assert pg == [5, 59] and cite.marker(pg) == " [tr. 5, 59]" and cite.marker([]) == ""


def test_verify_rendered_flags_foreign_pages():
    assert cite.verify_rendered("x [tr. 5] y [tr. 5, 59]", EVS) == []
    assert cite.verify_rendered("x [tr. 5, 123]", EVS) == [123]


def test_empty_status_and_plan_level_problems():
    assert not cite.check(plan(claims=[]), EVS, {}, "").ok
    assert cite.check(plan(status="not_in_report", claims=[]), EVS, {}, "").ok


@pytest.mark.parametrize("args,status,reason", [
    (("ambiguous", 0, 0, [], "Ý bạn là năm nào?", False), "ambiguous", "llm_ambiguous"),
    (("ambiguous", 0, 0, [], None, False), "ambiguous", "llm_ambiguous"),
    (("not_in_report", 0, 0, [], None, False), "not_in_report", "llm_declined"),
    (("answered", 2, 0, [], None, False), "not_in_report", "all_claims_invalid"),
    (("answered", 0, 0, [], None, False), "not_in_report", "no_claims"),
    (("answered", 2, 1, [], None, False), "partial", "claims_dropped"),
    (("answered", 1, 1, ["C1: chia cho 0"], None, False), "partial", "claims_dropped"),
    (("partial", 1, 1, [], None, False), "partial", "partial"),
    (("answered", 2, 2, [], None, False), "answered", "grounded"),
])
def test_decide(args, status, reason):
    d = refuse.decide(*args)
    assert (d.status, d.reason) == (status, reason)
    if status in ("ambiguous", "not_in_report"): assert d.notes


def test_refusal_never_claims_content():
    d = refuse.decide("answered", 3, 0, [], None, True)
    assert d.status == "not_in_report" and refuse.MSG_NOT_IN_REPORT in d.notes

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


def test_bracketed_ids_from_the_model_are_normalised():
    from tcb import plan
    p = plan.parse('{"status":"answered","claims":[{"text_vi":"x {C1}","evidence":["[E4]"],"calc_refs":["{C1}"]}],'
                   '"calculations":[{"cid":"[C1]","op":"growth_pct","inputs":["[Q33]","Q34"]}],'
                   '"literals":[{"lid":"[L1]","evidence":"[E2]","number":"1","unit":"","period":""}]}')
    assert p.claims[0].evidence == ["E4"] and p.claims[0].calc_refs == ["C1"] and p.calculations[0].cid == "C1"
    assert p.calculations[0].inputs == ["Q33", "Q34"] and p.literals[0].lid == "L1" and p.literals[0].evidence == "E2"


MAG = cite.evidence_from([node("k", "Tổng tài sản | 1.192 | Nghìn tỷ đồng | Tăng 21,8% so với | năm trước", 5, "kpi"),
                          node("t", "TỔNG TÀI SẢN | 31/12/2025 Triệu đồng: 1.192.344.137; 31/12/2024 Triệu đồng: 978.798.549", 297, "table"),
                          node("n", "Ngân hàng phục vụ hơn 18 triệu khách hàng qua 302 chi nhánh", 4),
                          node("r", "[Khối Ngân hàng Bán lẻ (RBG)] Dư nợ vay | 328,1 | Nghìn | tỷ đồng | 26,9% N/N", 59, "kpi")])


@pytest.mark.parametrize("text,eid,ok", [
    ("Tổng tài sản đạt 1.192 nghìn tỷ đồng, tăng 21,8%.", "E1", True),
    ("Tổng tài sản đạt 1.192 tỷ đồng.", "E1", False),                      # wrong by a factor of 1,000
    ("Tổng tài sản đạt 1.192.", "E1", False),                              # magnitude dropped
    ("Tổng tài sản đạt 1.192.344.137 triệu đồng.", "E2", True),            # table: unit printed in the column header
    ("Tổng tài sản đạt 1.192.344.137 tỷ đồng.", "E2", False),
    ("Hơn 18 triệu khách hàng, 302 chi nhánh.", "E3", True),
    ("Hơn 18 nghìn khách hàng.", "E3", False),
    ("Dư nợ vay 328,1 nghìn tỷ đồng.", "E4", True),                    # tile split across lines: 'Nghìn | tỷ đồng'
    ("Dư nợ vay 328,1 tỷ đồng.", "E4", False),
])
def test_magnitude_must_match_the_evidence(text, eid, ok):
    ck = cite.check(plan(claims=[{"text_vi": text, "evidence": [eid]}]), MAG, {}, "")
    assert ck.ok == ok, ck.flat


@pytest.mark.parametrize("texts,dups", [
    (["Có 302 chi nhánh.", "Hiện diện tại 29 trên 34 tỉnh.", "Mạng lưới gồm 302 chi nhánh tại 29 tỉnh."], {2}),
    (["Tổng tài sản là 1.192 nghìn tỷ đồng, tăng 21,8%.", "Tổng tài sản năm 2025 là 1.192.344 tỷ đồng."], {1}),   # rounding
    (["Tổng thu nhập hoạt động là 53,4 nghìn tỷ đồng.", "Theo bảng là 53.391 tỷ đồng.", "CAGR 2018–2025 là 16,5%."], {1}),
    (["CASA đạt 40,4%.", "CASA đạt 40,4% nhờ số dư CASA tăng 17%."], set()),                # adds a new figure: kept
    (["Lợi nhuận 32,5 nghìn tỷ đồng.", "Báo cáo khác ghi 31,9 nghìn tỷ đồng."], set()),     # a real disagreement: kept
    (["CASA là 40,4%.", "RBG là Khối Ngân hàng Bán lẻ."], set()),                          # no figures: kept
    (["Tỷ lệ 40%.", "Tăng 40 điểm."], set()),                                              # % and a count differ
])
def test_duplicate_claims(texts, dups):
    assert cite.duplicate_claims(texts) == dups


def test_decimal_comma_never_matches_thousands_dot():
    """'179,501' is 179.5; the evidence '179.501' is one hundred seventy-nine thousand. They must not match."""
    evs = cite.evidence_from([node("t", "Tổng vốn chủ sở hữu (tỷ đồng) 2025 179.501", 256)])
    ck = cite.check(plan(claims=[{"text_vi": "Vốn chủ sở hữu là 179,501 nghìn tỷ đồng.", "evidence": ["E1"]}]), evs, {}, "")
    assert not ck.ok and any("179,501" in p for p in ck.flat)
    assert cite.check(plan(claims=[{"text_vi": "Vốn chủ sở hữu là 179.501 tỷ đồng.", "evidence": ["E1"]}]), evs, {}, "").ok


@pytest.mark.parametrize("text,value,shown", [
    ("Thu nhập giảm {C1} so với năm trước.", "-39.28", "Thu nhập giảm 39,3% so với năm trước."),
    ("Thu nhập tăng {C1} so với năm trước.", "-39.28", "Thu nhập giảm 39,3% so với năm trước."),
    ("Vốn tăng {C1}% trong năm.", "21.33", "Vốn tăng 21,3% trong năm."),
    ("Thu nhập giảm {C1}.", "13.19", "Thu nhập tăng 13,2%."),
])
def test_calculator_result_reads_correctly_in_the_sentence(text, value, shown):
    from tcb.calc import CalcResult
    r = CalcResult("growth_pct", D(value), "%", 1, 1, ("Q1", "Q2"), "")
    assert cite.render_claim(P.Claim.model_validate({"text_vi": text, "evidence": ["E1"]}), {"C1": r}) == shown


def test_unit_written_after_a_result_is_not_doubled():
    from tcb.calc import CalcResult
    r = CalcResult("sum", D(12122934), "VND", 10**6, 0, ("Q1", "Q2"), "")
    cl = P.Claim.model_validate({"text_vi": "Tổng cộng {C1} triệu đồng.", "evidence": ["E1"]})
    assert cite.render_claim(cl, {"C1": r}) == "Tổng cộng 12.122.934 triệu đồng."


def test_numbers_in_the_question_are_premises_not_evidence():
    evs = cite.evidence_from([node("a", "Tổng tài sản đạt 100 tỷ đồng năm 2025", 5)])
    q = "Tổng tài sản năm 2025 là 999 tỷ đồng phải không?"
    assert not cite.check(plan(claims=[{"text_vi": "Tổng tài sản năm 2025 là 999 tỷ đồng.", "evidence": ["E1"]}]), evs, {}, q).ok
    assert cite.check(plan(claims=[{"text_vi": "Tổng tài sản năm 2025 là 100 tỷ đồng.", "evidence": ["E1"]}]), evs, {}, q).ok


@pytest.mark.parametrize("text,ok", [
    ("Lợi nhuận là -100 triệu đồng.", False),       # sign not in the evidence
    ("Lợi nhuận là 100 triệu đồng.", True),
    ("Chi phí là 5 tỷ đồng.", False),                # small integers with a unit are figures, not list counts
    ("Có 3 yếu tố chính.", True),
])
def test_sign_and_small_figures(text, ok):
    evs = cite.evidence_from([node("a", "Lợi nhuận đạt 100 triệu đồng năm 2025", 5)])
    assert cite.check(plan(claims=[{"text_vi": text, "evidence": ["E1"]}]), evs, {}, "").ok == ok


@pytest.mark.parametrize("lit,ok", [
    ({"number": "100", "unit": "triệu đồng", "period": "2025"}, True),
    ({"number": "100", "unit": "tỷ đồng", "period": "2025"}, False),     # unit not the one printed next to the number
    ({"number": "100", "unit": "triệu đồng", "period": "2030"}, False),  # year in neither the block nor the question
    ({"number": "101", "unit": "triệu đồng", "period": "2025"}, False),
])
def test_literal_unit_and_period_are_checked_against_the_block(lit, ok):
    evs = cite.evidence_from([node("a", "Lợi nhuận đạt 100 triệu đồng năm 2025", 5)])
    pl = plan(literals=[{"lid": "L1", "evidence": "E1", **lit}])
    got, probs = cite.literal_quantities(pl, evs, "Lợi nhuận?")
    assert ("L1" in got) == ok, probs

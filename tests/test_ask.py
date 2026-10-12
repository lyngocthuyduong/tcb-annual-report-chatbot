"""INTEGRATION on the REAL parsed PDF (nodes/pages/figures are real), but retrieval is a fixed-node stand-in and the
LLM is a scripted stand-in. Proves pipeline + citation + refusal + calculator wiring. It does NOT prove retrieval
quality, nor that any real LLM produces such plans (see tests/test_live.py and the eval runner)."""
import pytest

from fakes import NodeSet, Scripted, eid_of, qid_of
from tcb import ask, chat, llm

pytestmark = pytest.mark.integ


@pytest.fixture(scope="module")
def N(doc):
    def pick(pred):
        return next(n for n in doc["nodes"] if pred(n))
    return {
        "b02": pick(lambda n: n.content_type == "table" and "TỔNG TÀI SẢN" in n.text and n.printed_page_number == 297),
        "kpi_assets": pick(lambda n: n.content_type == "kpi" and "Tổng tài sản" in n.text and n.printed_page_number == 5),
        "kpi_casa": pick(lambda n: n.content_type == "kpi" and "Tỷ lệ CASA" in n.text and n.printed_page_number == 5),
        "gl_casa": pick(lambda n: n.content_type == "glossary" and "] CASA:" in n.text),
        "hr": pick(lambda n: n.content_type == "narrative" and "12.705 người" in n.text and n.printed_page_number == 52),
    }


def engine(N, handler, keys=("b02", "kpi_assets", "kpi_casa", "gl_casa")):
    return ask.Engine(NodeSet([N[k] for k in keys]), Scripted(handler)), None


def run(N, handler, q="Tổng tài sản?", turns=None, keys=("b02", "kpi_assets", "kpi_casa", "gl_casa")):
    e = ask.Engine(NodeSet([N[k] for k in keys]), Scripted(handler))
    return e.ask(q, turns), e.prov


def test_real_page_mapping_and_figures(N):
    assert N["b02"].printed_page_number == 297 and N["kpi_casa"].printed_page_number == 5
    assert N["gl_casa"].printed_page_number == 386 and "Tiền gửi không kỳ hạn" in N["gl_casa"].text
    assert "1.192.344.137" in N["b02"].text and "978.798.549" in N["b02"].text


def test_figure_answer_cites_real_printed_page(N):
    h = lambda rq, i: {"status": "answered", "claims": [{
        "text_vi": "Tổng tài sản tại 31/12/2025 là 1.192.344.137 triệu đồng.", "evidence": [eid_of(rq, "1.192.344.137")]}]}
    a, p = run(N, h)
    assert a.status == "answered" and a.answer.endswith("[tr. 297]") and len(p.calls) == 1
    assert a.citations[0]["page"] == 297 and a.citations[0]["pdf_page"] == 149 and a.provider == "scripted"
    assert a.trace["llm"]["calls"] == 1 and a.trace["citation_leak"] == []


def test_kpi_and_glossary_pages(N):
    h = lambda rq, i: {"status": "answered", "claims": [
        {"text_vi": "Tỷ lệ CASA là 40,4%.", "evidence": [eid_of(rq, "Tỷ lệ CASA | 40,4%")]},
        {"text_vi": "CASA là Tiền gửi không kỳ hạn.", "evidence": [eid_of(rq, "CASA: Tiền gửi")]}]}
    a, _ = run(N, h, "CASA là gì và tỷ lệ bao nhiêu?")
    assert a.answer.splitlines()[0].endswith("[tr. 5]") and a.answer.splitlines()[1].endswith("[tr. 386]")


def test_calculation_is_deterministic_and_cited(N):
    def h(rq, i):
        return {"status": "answered",
                "calculations": [{"cid": "C1", "op": "growth_pct",
                                  "inputs": [qid_of(rq, "TỔNG TÀI SẢN", "31/12/2025"), qid_of(rq, "TỔNG TÀI SẢN", "31/12/2024")]}],
                "claims": [{"text_vi": "Tổng tài sản tăng {C1} so với cuối năm 2024.", "evidence": [eid_of(rq, "TỔNG TÀI SẢN")], "calc_refs": ["C1"]}]}
    a, _ = run(N, h, "Tổng tài sản tăng bao nhiêu phần trăm?")
    assert a.status == "answered" and "tăng 21,8% so với" in a.answer and a.answer.endswith("[tr. 297]")
    assert a.trace["calcs"]["C1"]["formula"].startswith("(Q") and a.trace["calc_errors"] == []


def _lit_plan(rq, n2024="11.848"):
    e = eid_of(rq, "12.705 người")
    return {"status": "answered",
            "literals": [{"lid": "L1", "evidence": e, "number": "12.705", "unit": "người", "period": "2025"},
                         {"lid": "L2", "evidence": e, "number": n2024, "unit": "người", "period": "2024"}],
            "calculations": [{"cid": "C1", "op": "diff", "inputs": ["L1", "L2"]}],
            "claims": [{"text_vi": "Tổng số nhân sự năm 2025 tăng thêm {C1} người so với năm 2024.", "evidence": [e], "calc_refs": ["C1"]}]}


def test_calculation_on_narrative_numbers_via_literals(N):
    """Numbers that live in text (no table quantity) reach the calculator only if they occur verbatim in the cited block."""
    a, p = run(N, lambda rq, i: _lit_plan(rq), "Nhân sự năm 2025 tăng thêm bao nhiêu người so với năm 2024?", keys=("hr", "b02"))
    assert a.status == "answered" and "tăng thêm 857 người" in a.answer and a.answer.endswith("[tr. 52]") and len(p.calls) == 1
    assert a.trace["literals"]["L1"]["page"] == 52 and a.trace["calc_errors"] == []


def test_invented_literal_is_rejected(N):
    a, p = run(N, lambda rq, i: _lit_plan(rq, "11.000"), "Nhân sự năm 2025 tăng thêm bao nhiêu người so với năm 2024?", keys=("hr", "b02"))
    assert a.status != "answered" and "857" not in a.answer and len(p.calls) == 2       # one repair, then refused
    assert any("L2: number 11.000 not found" in x for x in a.trace["check_problems"])


def test_llm_arithmetic_is_rejected(N):
    """Model writes its own (wrong) percentage instead of using the calculator -> claim dropped after 1 retry."""
    h = lambda rq, i: {"status": "answered", "claims": [{"text_vi": "Tổng tài sản tăng 25,0%.", "evidence": [eid_of(rq, "TỔNG TÀI SẢN")]}]}
    a, p = run(N, h)
    assert a.status == "not_in_report" and a.trace["decision"]["reason"] == "all_claims_invalid" and len(p.calls) == 2
    assert "25,0" not in a.answer and a.citations == []


def test_hallucinated_page_and_unknown_evidence(N):
    h = lambda rq, i: {"status": "answered", "claims": [{"text_vi": "CASA là 40,4% [tr. 999].", "evidence": [eid_of(rq, "40,4%")]},
                                                         {"text_vi": "CASA là 40,4%.", "evidence": ["E77"]}]}
    a, _ = run(N, h)
    assert a.status == "not_in_report" and "999" not in a.answer


def test_partial_keeps_valid_claims_and_drops_bad(N):
    h = lambda rq, i: {"status": "answered", "claims": [
        {"text_vi": "Tỷ lệ CASA là 40,4%.", "evidence": [eid_of(rq, "Tỷ lệ CASA | 40,4%")]},
        {"text_vi": "ROE là 99,9%.", "evidence": [eid_of(rq, "Tỷ lệ CASA | 40,4%")]}]}
    a, _ = run(N, h)
    assert a.status == "partial" and "40,4%" in a.answer and "99,9" not in a.answer and "[tr. 5]" in a.answer
    assert "không thể đối chiếu" in a.answer


def test_not_in_report_has_no_citation(N):
    a, p = run(N, lambda rq, i: {"status": "not_in_report", "claims": []}, "Giá cổ phiếu Vinfast hôm nay?")
    assert a.status == "not_in_report" and a.citations == [] and "[tr." not in a.answer and len(p.calls) == 1


def test_ambiguous_returns_clarification(N):
    a, _ = run(N, lambda rq, i: {"status": "ambiguous", "claims": [], "clarification_vi": "Bạn hỏi năm nào?"}, "Tăng bao nhiêu?")
    assert a.status == "ambiguous" and a.answer == "Bạn hỏi năm nào?"


def test_calc_error_is_reported_not_guessed(N):
    def h(rq, i):
        q = qid_of(rq, "TỔNG TÀI SẢN", "31/12/2025")
        return {"status": "answered", "calculations": [{"cid": "C1", "op": "growth_pct", "inputs": [q, q]}],
                "claims": [{"text_vi": "Tăng {C1}.", "evidence": [eid_of(rq, "TỔNG TÀI SẢN")], "calc_refs": ["C1"]}]}
    a, _ = run(N, h)
    assert a.status == "not_in_report" and "Không thể tính" in a.answer and "C1" in a.answer
    assert a.trace["calc_errors"]


def test_provider_failure_is_surfaced_not_swapped(N):
    a, p = run(N, lambda rq, i: llm.ProviderTimeout("scripted", "scripted-1", "slow"))
    assert a.status == "error" and "scripted/scripted-1" in a.answer and "timeout" in a.answer and len(p.calls) == 1
    assert "Không tự động chuyển" in a.answer


def test_missing_model_snapshot_is_an_error_not_a_crash(N):
    class Boom(NodeSet):
        def search(self, q): raise FileNotFoundError("/models/bge-m3 missing")
    e = ask.Engine(Boom([]), Scripted(lambda rq, i: {}))
    a = e.ask("x?")
    assert a.status == "error" and "bge-m3" in a.answer


def test_empty_question(N):
    a, p = run(N, lambda rq, i: {}, "   ")
    assert a.status == "ambiguous" and p.calls == []


# ---------------- conversation ----------------
PREV = [chat.Turn("Tổng tài sản năm 2025?", "Tổng tài sản của Techcombank tại 31/12/2025 là bao nhiêu?", "1.192.344.137 triệu đồng", "answered")]


def test_followup_is_rewritten_then_answered(N):
    def h(rq, i):
        if i == 0:
            assert "CÂU HỎI CUỐI: Còn năm 2024?" in rq.user
            return {"standalone": "Tổng tài sản của Techcombank tại 31/12/2024 là bao nhiêu?"}
        assert "31/12/2024" in rq.user and "Ngữ cảnh hội thoại" in rq.user
        return {"status": "answered", "claims": [{"text_vi": "Tổng tài sản tại 31/12/2024 là 978.798.549 triệu đồng.", "evidence": [eid_of(rq, "978.798.549")]}]}
    a, p = run(N, h, "Còn năm 2024?", PREV)
    assert a.trace["rewrite_mode"] == "llm" and "2024" in a.standalone and a.status == "answered" and len(p.calls) == 2
    assert a.answer.endswith("[tr. 297]")


def test_unfaithful_rewrite_degrades_to_concat_and_says_so(N):
    def h(rq, i):
        if i == 0: return {"standalone": "Tổng tài sản là bao nhiêu?"}          # dropped "2024"
        return {"status": "not_in_report", "claims": []}
    a, _ = run(N, h, "Còn năm 2024?", PREV)
    assert a.trace["rewrite_mode"] == "concat" and "2024" in a.standalone


def test_standalone_question_is_not_rewritten(N):
    seen = []
    a, p = run(N, lambda rq, i: (seen.append(rq.user), {"status": "not_in_report", "claims": []})[1],
               "Techcombank công bố tổng tài sản năm 2025 ở bảng nào của báo cáo tài chính hợp nhất?", PREV)
    assert a.trace["rewrite_mode"] == "none" and len(p.calls) == 1


def test_session_keeps_last_turns():
    s = chat.Session()
    for i in range(7): s.add(chat.Turn(str(i), str(i), "a", "answered"))
    assert [t.question for t in s.turns] == ["3", "4", "5", "6"]


def test_prompt_budget_trims_lowest_ranked_evidence_but_keeps_best(N, monkeypatch):
    from tcb import conf
    monkeypatch.setattr(conf, "PROMPT_EVIDENCE_CHARS", 100)
    a, p = run(N, lambda rq, i: {"status": "not_in_report", "claims": []})
    assert len(a.trace["evidence"]) == 1 and a.trace["evidence"][0]["page"] == 297        # best-ranked kept

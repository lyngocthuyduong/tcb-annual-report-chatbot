"""UNIT: HTTP layer with a scripted LLM and synthetic nodes (no PDF, no models). Proves routing/status codes/sessions only."""
import json
from types import SimpleNamespace as NS

import pytest
from fastapi.testclient import TestClient

from fakes import NodeSet, Scripted, eid_of
from tcb import api, ask, llm, smoke

pytestmark = pytest.mark.unit

NODE = {"chunk_id": "n1", "text": "Tỷ lệ CASA | 40,4%", "printed_page_number": 5, "pdf_page_index": 3, "side": "R",
        "section": "Chúng tôi là ai", "content_type": "kpi", "page_number_source": "footer"}


class RT(NodeSet):
    ix = NS(manifest={"counts": {"nodes": 1}, "embedding": {"model": "TEST"}, "built_at": "t"}, nodes={"n1": NODE})


def client(handler, tmp_path, pdf=None):
    e = ask.Engine(RT([NODE]), Scripted(handler))
    return TestClient(api.create_app(e, tmp_path / "tr", pdf or tmp_path / "none.pdf")), e


OK = lambda rq, i: {"status": "answered", "claims": [{"text_vi": "CASA là 40,4%.", "evidence": [eid_of(rq, "40,4%")]}]}


def test_health_and_ui(tmp_path):
    c, _ = client(OK, tmp_path)
    j = c.get("/health").json()
    assert j["status"] == "ok" and j["provider"] == "scripted" and j["index"]["nodes"] == 1
    r = c.get("/"); assert r.status_code == 200 and "<form" in r.text and "/chat" in r.text


def test_chat_answer_has_citation_and_trace_file(tmp_path):
    c, _ = client(OK, tmp_path)
    r = c.post("/chat", json={"question": "CASA bao nhiêu?"}); j = r.json()
    assert r.status_code == 200 and j["status"] == "answered" and j["answer"].endswith("[tr. 5].") and j["session_id"]
    assert j["citations"][0]["pdf_page"] == 3 and "trace" not in j
    rec = json.loads((tmp_path / "tr" / "chat.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert rec["trace"]["llm"]["calls"] == 1 and rec["trace"]["decision"]["reason"] == "grounded"


def test_session_gives_followup_context(tmp_path):
    seen = []
    def h(rq, i):
        seen.append(rq.user)
        if "CÂU HỎI CUỐI" in rq.user: return {"standalone": "Tỷ lệ nợ xấu của Techcombank năm 2025 là bao nhiêu?"}
        return OK(rq, i)
    c, _ = client(h, tmp_path)
    sid = c.post("/chat", json={"question": "CASA bao nhiêu?"}).json()["session_id"]
    j = c.post("/chat", json={"question": "Còn nợ xấu?", "session_id": sid}).json()
    assert j["standalone"].startswith("Tỷ lệ nợ xấu") and any("CÂU HỎI CUỐI: Còn nợ xấu?" in s for s in seen)
    j2 = c.post("/chat", json={"question": "Còn nợ xấu?"}).json()          # new session: no history -> no rewrite
    assert j2["standalone"] == "Còn nợ xấu?"


@pytest.mark.parametrize("exc,code", [(llm.ProviderTimeout("scripted", "m", "t"), 504), (llm.ProviderUnavailable("scripted", "m", "u"), 503),
                                      (llm.ProviderRejected("scripted", "m", "r"), 502)])
def test_provider_errors_map_to_http_and_name_provider(tmp_path, exc, code):
    c, _ = client(lambda rq, i: exc, tmp_path)
    r = c.post("/chat", json={"question": "x?"}); j = r.json()
    assert r.status_code == code and j["status"] == "error" and "scripted/m" in j["answer"] and j["error_kind"] == exc.kind


def test_validation_and_evidence_and_pdf(tmp_path):
    pdf = tmp_path / "a.pdf"; pdf.write_bytes(b"%PDF-1.4 x")
    c, _ = client(OK, tmp_path, pdf)
    assert c.post("/chat", json={"question": ""}).status_code == 422
    assert c.post("/chat", json={"question": "x" * 2001}).status_code == 422
    assert c.get("/evidence/n1").json()["printed_page_number"] == 5 and c.get("/evidence/zzz").status_code == 404
    assert c.get("/pdf").headers["content-type"] == "application/pdf"
    c2, _ = client(OK, tmp_path)
    assert c2.get("/pdf").status_code == 404


# ---------------- smoke command: both honest branches ----------------
def test_smoke_not_executed_when_ollama_unreachable(tmp_path, capsys):
    rc = smoke.run(str(tmp_path / "s.json"), {"OLLAMA_BASE_URL": "http://127.0.0.1:9", "LLM_TIMEOUT_S": "2"})
    out = capsys.readouterr().out; j = json.loads((tmp_path / "s.json").read_text(encoding="utf-8"))
    assert rc == 6 and out.startswith("Ollama smoke test: Not Executed. Reason: Ollama unreachable") and j["executed"] is False


def test_smoke_executed_branch_logic_with_stand_in(tmp_path, capsys, monkeypatch):
    good = lambda rq, i: {"status": "answered", "claims": [{"text_vi": "Tỷ lệ CASA là 40,4%.", "evidence": ["E1"]}]}
    monkeypatch.setattr(llm, "get", lambda env=None: Scripted(good))
    rc = smoke.run(str(tmp_path / "s.json"))
    out = capsys.readouterr().out
    assert rc == 0 and out.startswith("Ollama smoke test: Executed. Setup: model=qwen2.5:7b, latency=") and "quality=pass" in out

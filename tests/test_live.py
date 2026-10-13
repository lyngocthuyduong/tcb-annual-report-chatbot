"""Live tests: real models, real providers, the shipped index. They need keys or a running service, so they run on
demand: pytest -m live (or live_models, live_anthropic, live_openai, live_ollama). A test whose service is missing skips.
The Anthropic path they cover is the one smoke.sh and the eval runs in runs/ exercise end to end."""
import os
from pathlib import Path

import pytest

from tcb import conf

ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.live_models
def test_real_models_load_and_behave():
    if not (conf.MODELS_DIR / "bge-m3").exists():
        pytest.skip(f"{conf.MODELS_DIR} has no baked models (run inside the Docker image)")
    from tcb import cli
    assert cli.cmd_check(None) == 0                     # same gate as `tcb check`: dim 1024, revision matches, related text closer than unrelated


@pytest.mark.live_openai
def test_real_openai_returns_valid_plan():
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")
    from tcb import cite, gen, llm, plan as P, smoke
    prov = llm.get({"PROVIDER": "openai", "OPENAI_API_KEY": os.environ["OPENAI_API_KEY"],
                    "OPENAI_MODEL": os.getenv("OPENAI_MODEL", "gpt-4o-mini")})
    evs = cite.evidence_from([{"chunk_id": k, "text": v, "printed_page_number": 5, "pdf_page_index": 3, "side": "right",
                               "section": "", "content_type": "kpi"} for k, v in smoke.EVIDENCE])
    g = gen.generate_plan(prov, P.SYSTEM, P.user_prompt(smoke.QUESTION, smoke.EVIDENCE, []),
                          check=lambda pl: cite.check(pl, evs, {}, smoke.QUESTION).flat)
    assert g.plan.status == "answered" and not g.problems and any("40,4" in c.text_vi for c in g.plan.claims)


@pytest.mark.live_anthropic
def test_real_anthropic_returns_valid_plan():
    if not os.getenv("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")
    from tcb import cite, gen, llm, plan as P, smoke
    prov = llm.get({"PROVIDER": "anthropic", "ANTHROPIC_API_KEY": os.environ["ANTHROPIC_API_KEY"]})
    evs = cite.evidence_from([{"chunk_id": k, "text": v, "printed_page_number": 5, "pdf_page_index": 3, "side": "right",
                               "section": "", "content_type": "kpi"} for k, v in smoke.EVIDENCE])
    g = gen.generate_plan(prov, P.SYSTEM, P.user_prompt(smoke.QUESTION, smoke.EVIDENCE, []),
                          check=lambda pl: cite.check(pl, evs, {}, smoke.QUESTION).flat)
    assert g.plan.status == "answered" and not g.problems


@pytest.mark.live_ollama
def test_real_ollama_smoke(tmp_path):
    import json
    from tcb import smoke
    rc = smoke.run(str(tmp_path / "s.json"))
    j = json.loads((tmp_path / "s.json").read_text(encoding="utf-8"))
    if not j["executed"]:
        pytest.skip(f"Ollama could not be reached: {j['reason']}")
    assert rc == 0, f"Ollama ran but quality={j['quality']}"


@pytest.mark.live
def test_end_to_end_shipped_index_real_models_real_provider():
    """Shipped idx/ + baked models + configured PROVIDER. The only test that exercises the full submission path."""
    idx = ROOT / "idx"
    if not (idx / "manifest.json").exists():
        pytest.skip("idx/ not built yet (needs the real BGE-M3 -> see the development runbook in README)")
    if not (conf.MODELS_DIR / "bge-m3").exists():
        pytest.skip("baked models missing")
    from tcb.boot import load_engine
    e = load_engine(index_dir=idx)
    a = e.ask("Tỷ lệ CASA của Techcombank năm 2025 là bao nhiêu?")
    import re   # 40,4% is stated on printed pp. 5 (highlights), 49 and 53 (verified in the PDF text): any of them is a correct citation
    pages = {int(n) for m in re.findall(r"\[tr\. ([\d, ]+)\]", a.answer) for n in re.findall(r"\d+", m)}
    assert a.status == "answered" and "40,4" in a.answer and pages & {5, 49, 53}, (a.status, a.error, a.answer)
    r = e.ask("Techcombank dự báo lợi nhuận trước thuế năm 2027 là bao nhiêu?")
    assert r.status == "not_in_report" and "[tr." not in r.answer

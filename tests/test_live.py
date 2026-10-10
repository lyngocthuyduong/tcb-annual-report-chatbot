"""LIVE tests: real models / real providers / shipped index. Opt-in:  pytest -m live  (or live_models / live_openai / live_ollama).
A SKIP is not a pass. These have NOT been run in the authoring sandbox (no HF/PyTorch/Docker/Ollama/OpenAI access there)."""
import os
from pathlib import Path

import pytest

from tcb import conf

ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.live_models
def test_real_models_load_and_behave():
    if not (conf.MODELS_DIR / "bge-m3").exists() or not (conf.MODELS_DIR / "bge-reranker-v2-m3").exists():
        pytest.skip(f"{conf.MODELS_DIR} has no baked models (run inside the Docker image)")
    from tcb import cli
    assert cli.cmd_check(None) == 0                     # same gate as `tcb check`: dim 1024, revisions match, rerank pos > neg


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


@pytest.mark.live_openai
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
    j = json.loads((tmp_path / "s.json").read_text())
    if not j["executed"]:
        pytest.skip(f"Ollama smoke Not Executed: {j['reason']}")      # a skip, never a pass
    assert rc == 0, f"Ollama ran but quality={j['quality']}"


@pytest.mark.live
def test_end_to_end_shipped_index_real_models_real_provider():
    """Shipped idx/ + baked models + configured PROVIDER. The only test that exercises the full submission path."""
    idx = ROOT / "idx"
    if not (idx / "manifest.json").exists():
        pytest.skip("idx/ not built yet (needs the real BGE-M3 -> see README 'Build the index')")
    if not (conf.MODELS_DIR / "bge-m3").exists():
        pytest.skip("baked models missing")
    from tcb.boot import load_engine
    e = load_engine(index_dir=idx)
    a = e.ask("Tỷ lệ CASA của Techcombank năm 2025 là bao nhiêu?")
    assert a.status == "answered" and "40,4" in a.answer and "[tr. 5]" in a.answer
    r = e.ask("Techcombank dự báo lợi nhuận trước thuế năm 2027 là bao nhiêu?")
    assert r.status == "not_in_report" and "[tr." not in r.answer

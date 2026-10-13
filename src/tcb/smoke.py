"""Ollama smoke test (local model). Runs the real local model through the same plan/validation path on a tiny fixed
evidence set. Prints the standard log line; 'Not Executed' (with the reason) when Ollama is not reachable.
Only an actual run on a machine with Ollama proves the local path; mocks do not."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from . import cite, gen, llm, plan as P

EVIDENCE = [("E1", "[Thành tựu] Tỷ lệ CASA | 40,4%\nTiền gửi không kỳ hạn (CASA) chiếm 40,4% tổng tiền gửi khách hàng năm 2025."),
            ("E2", "[Thành tựu] Tỷ lệ nợ xấu | 1,13%")]
QUESTION = "Tỷ lệ CASA của Techcombank năm 2025 là bao nhiêu?"


def run(out: str = "out/smoke.json", env=None) -> int:
    e = dict(os.environ if env is None else env); e["PROVIDER"] = "ollama"
    model = e.get("OLLAMA_MODEL") or llm.DEFAULTS["ollama"]
    try:
        prov = llm.get(e)
        t = time.time()
        evs = cite.evidence_from([{"chunk_id": k, "text": v, "printed_page_number": 5, "pdf_page_index": 3, "side": "R",
                                   "section": "", "content_type": "kpi"} for k, v in EVIDENCE])
        g = gen.generate_plan(prov, P.SYSTEM, P.user_prompt(QUESTION, EVIDENCE, []), llm.max_tokens(e),
                              check=lambda pl: cite.check(pl, evs, {}, QUESTION).flat)
        dt = time.time() - t
    except llm.ProviderUnavailable as x:
        return _report(out, False, model, None, "n/a", f"Ollama unreachable: {x.msg}")
    except llm.LLMError as x:
        return _report(out, False, model, None, "n/a", f"{x.kind}: {x.msg}")
    ok = g.plan.status == "answered" and not g.problems and any("40,4" in c.text_vi for c in g.plan.claims)
    q = ("pass" if ok else "fail") + (" (needed repair retry)" if g.repaired else "")
    return _report(out, True, model, dt, q, "", g)


def _report(out, executed, model, dt, quality, reason, g=None) -> int:
    if executed:
        print(f"Ollama smoke test: Executed. Setup: model={model}, latency={dt:.1f}s, quality={quality}.")
    else:
        print(f"Ollama smoke test: Not Executed. Reason: {reason}. Setup: model={model}.")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps({"executed": executed, "model": model, "latency_s": dt and round(dt, 2), "quality": quality,
                                     "reason": reason, "plan": g.plan.model_dump() if g else None}, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0 if executed and quality.startswith("pass") else 6

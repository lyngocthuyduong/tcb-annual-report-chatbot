"""Answer pipeline: rewrite -> retrieve -> evidence/quantities -> LLM plan (<=1 repair) -> validate -> calculate
-> refusal policy -> render with citations. Every answer records provider+model; failures are surfaced, never swapped."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from . import calc, chat, cite, conf, find, gen, num, plan as P, refuse, vi
from .llm import LLMError, Provider


@dataclass
class Answer:
    question: str
    standalone: str
    status: str
    answer: str
    citations: list[dict] = field(default_factory=list)
    provider: str = ""
    model: str = ""
    error: str | None = None
    error_kind: str | None = None
    trace: dict = field(default_factory=dict)

    def to_dict(self, trace: bool = False) -> dict:
        d = {"question": self.question, "standalone": self.standalone, "status": self.status, "answer": self.answer,
             "citations": self.citations, "provider": self.provider, "model": self.model, "error": self.error,
             "error_kind": self.error_kind}
        if trace: d["trace"] = self.trace
        return d


def cost_usd(model: str, tin: int, tout: int) -> float | None:
    if model.startswith("qwen") or ":" in model: return 0.0       # local model
    p = conf.PRICES.get(model)
    return None if p is None else round((tin * p[0] + tout * p[1]) / 1e6, 6)


class Engine:
    def __init__(self, retriever: find.Retriever, provider: Provider, max_tokens: int = 1000):
        self.rt, self.prov, self.max_tokens = retriever, provider, max_tokens

    # ---------------------------------------------------------------- helpers
    def _quantities(self, nodes: list[dict], question: str, cap: int = 60) -> dict[str, num.Quantity]:
        """Table quantities of the evidence. Ten statement chunks can hold 200+ cells, so when over the cap the ones whose
        label shares the most words with the question are kept (then evidence order)."""
        k = [0]
        def nid():
            k[0] += 1; return f"Q{k[0]}"
        allq = [q for n in nodes for q in num.from_node(n, nid)]
        if len(allq) > cap:
            qt = set(vi.tokens(question))
            keep = sorted(range(len(allq)), key=lambda i: (-len(qt & set(vi.tokens(allq[i].label))), i))[:cap]
            allq = [allq[i] for i in sorted(keep)]
        return {q.qid: q for q in allq}

    def _retrieve(self, q: str, trace: dict):
        r = self.rt.search(q)
        trace["retrieval"] = {"query": r.query, "bm25": r.bm25, "dense": r.dense, "fused": r.fused[: self.rt.top_bm25],
                              "ranked": [[c, round(s, 5)] for c, s in r.ranked], "seconds": r.seconds}
        return r

    # ---------------------------------------------------------------- main
    def ask(self, question: str, turns: list[chat.Turn] | None = None) -> Answer:
        t0 = time.time()
        q = question.strip()
        tr: dict = {"question": q, "provider": self.prov.name, "model": self.prov.model}
        ans = Answer(q, q, "error", "", provider=self.prov.name, model=self.prov.model, trace=tr)
        calls = []
        try:
            if not q:
                ans.status, ans.answer = "ambiguous", refuse.MSG_AMBIGUOUS
                return ans
            standalone, mode, rr = chat.rewrite(self.prov, q, turns or [], 300)
            if rr: calls.append(rr)
            ans.standalone = standalone
            tr.update(standalone=standalone, rewrite_mode=mode)

            res = self._retrieve(standalone, tr)
            nodes, used = [], 0
            for n in self.rt.top_nodes(res):                # rank order; stop at the prompt budget (always keep the best one)
                if nodes and used + len(n["text"]) > conf.PROMPT_EVIDENCE_CHARS: break
                nodes.append(n); used += len(n["text"])
            tr["evidence_chars"] = used
            evs = cite.evidence_from(nodes)
            qmap = self._quantities(nodes, standalone)
            tr["evidence"] = [{"eid": e.eid, "chunk_id": e.chunk_id, "page": e.page, "pdf_page": e.pdf_page,
                               "type": e.content_type, "source": e.page_source} for e in evs]
            tr["quantities"] = {k: {"label": v.label, "value": str(v.value), "dim": v.dim, "scale": v.scale,
                                    "period": v.period.label, "page": v.page} for k, v in qmap.items()}

            qlines = [f"[{k}] {v.label} | {v.period.label or 'kỳ không rõ'} | {num.fmt(v.value)} {num.unit_vi(v.dim, v.scale)}"
                      for k, v in qmap.items()]
            user = P.user_prompt(standalone, [(e.eid, e.text) for e in evs], qlines, chat.history_text(turns or []) if turns else "")
            g = gen.generate_plan(self.prov, P.SYSTEM, user, self.max_tokens,
                                  check=lambda pl: cite.check(pl, evs, qmap, standalone).flat)
            calls += g.responses
            plan = g.plan
            tr["plan"] = plan.model_dump(); tr["repaired"] = g.repaired

            lits, _ = cite.literal_quantities(plan, evs)
            tr["literals"] = {k: {"value": str(v.value), "dim": v.dim, "period": v.period.label, "page": v.page} for k, v in lits.items()}
            qmap = {**qmap, **lits}
            ck = cite.check(plan, evs, qmap, standalone)
            tr["check_problems"] = ck.flat
            # ---- calculations (deterministic)
            results: dict[str, calc.CalcResult] = {}
            cerr: dict[str, str] = {}
            for c in plan.calculations:
                try:
                    results[c.cid] = calc.run(c.op, [qmap[x] for x in c.inputs if x in qmap], c.to)
                except calc.CalcError as e:
                    cerr[c.cid] = f"{c.cid} ({c.op}): {e}"
            tr["calcs"] = {k: {"op": v.op, "inputs": list(v.inputs), "value": str(v.value), "text": v.text,
                               "formula": v.formula} for k, v in results.items()}
            # ---- keep only grounded claims whose calculations succeeded
            kept, used_err = [], []
            for i, cl in enumerate(plan.claims):
                if i in ck.per_claim: continue
                refs = set(cite._PLACEHOLDER.findall(cl.text_vi)) | set(cl.calc_refs)
                bad = [r for r in refs if r in cerr]
                if bad: used_err += [cerr[r] for r in bad]; continue
                kept.append(cl)
            tr["calc_errors"] = used_err
            d = refuse.decide(plan.status, len(plan.claims), len(kept), used_err, plan.clarification_vi, bool(g.problems))
            tr["decision"] = {"status": d.status, "reason": d.reason}
            # ---- claims that restate figures already given are only flagged: removing them (runs dev9/val9) also removed
            # remarks they carried ("dẫn đầu hệ thống"), so one statement per figure is left to the prompt (rule 9)
            dups = cite.duplicate_claims([cite.render_claim(cl, {}) if not (set(cite._PLACEHOLDER.findall(cl.text_vi)) | set(cl.calc_refs))
                                          else "" for cl in kept])
            tr["restated_figures"] = [kept[i].text_vi for i in sorted(dups)]

            # ---- render
            lines, cites = [], []
            for cl in kept:
                pg = cite.pages_of(cl, evs, plan, results, qmap)
                lines.append(cite.with_marker(cite.render_claim(cl, results), pg))
                for e in cl.evidence:
                    ev = next(x for x in evs if x.eid == e)
                    if not any(c["chunk_id"] == ev.chunk_id for c in cites):
                        cites.append({"eid": ev.eid, "page": ev.page, "pdf_page": ev.pdf_page, "section": ev.section,
                                      "chunk_id": ev.chunk_id, "type": ev.content_type, "snippet": cite.snippet(ev.text)})
            body = " ".join(lines)                                   # one paragraph: the claims are written to follow on
            if d.status == "partial" and plan.clarification_vi and not cite.check_text_numbers(plan.clarification_vi, evs, standalone):
                d.notes.insert(0, plan.clarification_vi.strip())
            ans.status = d.status
            ans.answer = "\n".join(x for x in [body] + d.notes if x)
            ans.citations = cites
            leak = cite.verify_rendered(ans.answer, evs, qmap)
            if leak:                                             # defence in depth; should be unreachable
                ans.status, ans.error = "error", f"citation outside retrieved evidence: {leak}"
            tr["citation_leak"] = leak
        except LLMError as e:
            ans.status, ans.error, ans.error_kind = "error", str(e), e.kind
            ans.answer = f"Không thể tạo câu trả lời vì lỗi nhà cung cấp LLM ({e.provider}/{e.model}, {e.kind}). Không tự động chuyển sang nhà cung cấp khác."
        except (FileNotFoundError, RuntimeError) as e:
            ans.status, ans.error, ans.error_kind = "error", f"{type(e).__name__}: {e}", "retrieval"
            ans.answer = "Không thể truy xuất: " + str(e)
        tin = sum(c.in_tokens or 0 for c in calls); tout = sum(c.out_tokens or 0 for c in calls)
        tr["llm"] = {"provider": self.prov.name, "model": self.prov.model, "calls": len(calls),
                     "in_tokens": tin or None, "out_tokens": tout or None,
                     "latency_s": round(sum(c.latency_s for c in calls), 3),
                     "cost_usd": cost_usd(self.prov.model, tin, tout) if calls else None}
        tr["total_s"] = round(time.time() - t0, 3)
        return ans

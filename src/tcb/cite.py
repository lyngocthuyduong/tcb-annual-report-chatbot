"""Evidence ids, citation construction and plan validation. Page numbers come ONLY from index metadata
(printed_page_number of the cited node); the LLM never supplies one."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from . import vi
from .calc import CalcResult
from .num import Quantity

_NUM = re.compile(r"\d+(?:[.,]\d+)*")
_PLACEHOLDER = re.compile(r"\{(C\d+)\}")
_PAGE_REF = re.compile(r"\[\s*tr\.?[^\]]*\]|\btrang\s+\d+|\bpage\s+\d+", re.I)
CITE_RE = re.compile(r"\[tr\. (\d+(?:, \d+)*)\]")


@dataclass(frozen=True)
class Ev:
    eid: str
    chunk_id: str
    text: str
    page: int | None            # printed page (what a human sees on the page)
    pdf_page: int               # 1-based PDF spread index
    side: str
    section: str
    content_type: str
    page_source: str            # footer | inferred


def evidence_from(nodes: list[dict]) -> list[Ev]:
    return [Ev(f"E{i}", n["chunk_id"], n["text"], n.get("printed_page_number"), n["pdf_page_index"], n["side"],
               n.get("section", ""), n["content_type"], n.get("page_number_source", ""))
            for i, n in enumerate(nodes, 1)]


def num_tokens(text: str) -> set[str]:
    """Numbers in canonical form, '%' dropped: '14,6%' -> '14.6', '1.192.344.137' -> '1192344137'."""
    out = set()
    for m in _NUM.finditer(vi.nfc(text)):
        t = m.group(0)
        out.add(vi.canon_num(t).rstrip("%"))
        out.add(t)
    return out


def _trivial(t: str) -> bool:
    return t.isdigit() and int(t) <= 12     # "3 yếu tố", "tháng 12", list counts


@dataclass
class Check:
    per_claim: dict[int, list[str]] = field(default_factory=dict)
    plan_level: list[str] = field(default_factory=list)

    @property
    def flat(self) -> list[str]:
        return self.plan_level + [f"claim {i + 1}: {p}" for i, ps in sorted(self.per_claim.items()) for p in ps]

    @property
    def ok(self) -> bool:
        return not self.plan_level and not self.per_claim


def literal_quantities(plan, evs: list[Ev]) -> tuple[dict[str, Quantity], list[str]]:
    """Quantities declared by the plan from evidence text. Kept only if the number occurs verbatim in the cited block."""
    from .num import NumError, literal
    by = {e.eid: e for e in evs}
    out, probs = {}, []
    for x in getattr(plan, "literals", []):
        e = by.get(x.evidence)
        if e is None:
            probs.append(f"{x.lid}: unknown evidence id {x.evidence}"); continue
        n = vi.nfc(x.number).strip()
        if n not in num_tokens(e.text) and vi.canon_num(n).rstrip("%") not in num_tokens(e.text):
            probs.append(f"{x.lid}: number {x.number} not found in {x.evidence}"); continue
        try:
            out[x.lid] = literal(x.lid, n, x.unit, x.period, x.label, e.chunk_id, e.page)
        except NumError as err:
            probs.append(f"{x.lid}: {err}")
    return out, probs


def check(plan, evs: list[Ev], quantities: dict[str, Quantity], question: str = "") -> Check:
    """Deterministic validation of an LLM plan against the retrieved evidence."""
    c = Check()
    lits, lprobs = literal_quantities(plan, evs)
    c.plan_level += lprobs
    quantities = {**quantities, **lits}
    eids = {e.eid: e for e in evs}
    cids = {x.cid for x in plan.calculations}
    q_nums = num_tokens(question)
    if plan.status in ("answered", "partial") and not plan.claims:
        c.plan_level.append(f"status={plan.status} but no claims")
    if len(cids) != len(plan.calculations):
        c.plan_level.append("duplicate calculation ids")
    for x in plan.calculations:
        for q in x.inputs:
            if q not in quantities:
                c.plan_level.append(f"{x.cid}: unknown quantity {q} (allowed: {', '.join(list(quantities)[:12]) or 'none'})")
    for i, cl in enumerate(plan.claims):
        ps: list[str] = []
        refs = set(_PLACEHOLDER.findall(cl.text_vi)) | set(cl.calc_refs)
        for r in sorted(refs):
            if r not in cids: ps.append(f"unknown calculation {r}")
        if not cl.evidence and not refs:
            ps.append("no evidence cited")
        for e in cl.evidence:
            if e not in eids: ps.append(f"unknown evidence id {e}")
            elif eids[e].page is None: ps.append(f"{e} has no printed page; cannot cite")
        if _PAGE_REF.search(cl.text_vi):
            ps.append("page reference written in text (the system adds citations)")
        cited_text = " ".join(eids[e].text for e in cl.evidence if e in eids)
        # quantities used by this claim's calculations count as evidence for the numbers they were read from
        for r in refs:
            for x in plan.calculations:
                if x.cid == r:
                    cited_text += " " + " ".join(quantities[q].raw for q in x.inputs if q in quantities)
        ok_nums = num_tokens(cited_text) | q_nums
        body = _PLACEHOLDER.sub(" ", cl.text_vi)
        bad = sorted({t for m in _NUM.finditer(vi.nfc(body)) for t in [m.group(0)]
                      if not _trivial(t) and t not in ok_nums and vi.canon_num(t).rstrip("%") not in ok_nums})
        if bad:
            ps.append("numbers not found in the cited evidence: " + ", ".join(bad[:6]))
        if ps: c.per_claim[i] = ps
    return c


def pages_of(claim, evs: list[Ev], plan, results: dict[str, CalcResult], quantities: dict[str, Quantity]) -> list[int]:
    by = {e.eid: e for e in evs}
    pg = {by[e].page for e in claim.evidence if e in by and by[e].page is not None}
    refs = set(_PLACEHOLDER.findall(claim.text_vi)) | set(claim.calc_refs)
    for x in plan.calculations:
        if x.cid in refs:
            pg |= {quantities[q].page for q in x.inputs if q in quantities and quantities[q].page is not None}
    return sorted(pg)


def render_claim(claim, results: dict[str, CalcResult]) -> str:
    return _PLACEHOLDER.sub(lambda m: results[m.group(1)].text if m.group(1) in results else m.group(0), claim.text_vi).strip()


def marker(pages: list[int]) -> str:
    return f" [tr. {', '.join(map(str, pages))}]" if pages else ""


def snippet(text: str, n: int = 240) -> str:
    t = re.sub(r"\s+", " ", text).strip()
    return t if len(t) <= n else t[:n].rstrip() + "…"


def verify_rendered(answer: str, evs: list[Ev], quantities: dict[str, Quantity] | None = None) -> list[str]:
    """Post-render guard (also used by eval): every [tr. N] must be a page of a retrieved evidence node."""
    allowed = {e.page for e in evs if e.page is not None}
    allowed |= {q.page for q in (quantities or {}).values() if q.page is not None}
    bad = []
    for m in CITE_RE.finditer(answer):
        for p in m.group(1).split(", "):
            if int(p) not in allowed: bad.append(int(p))
    return bad


def check_text_numbers(text: str, evs: list[Ev], question: str = "") -> list[str]:
    """Numbers in free text (e.g. the 'missing part' note of a partial answer) not present in ANY evidence."""
    ok = num_tokens(" ".join(e.text for e in evs)) | num_tokens(question)
    return sorted({m.group(0) for m in _NUM.finditer(vi.nfc(text))
                   if not _trivial(m.group(0)) and m.group(0) not in ok and vi.canon_num(m.group(0)).rstrip("%") not in ok})

"""Evidence ids, citation construction and plan validation. Page numbers come only from index metadata
(printed_page_number of the cited node); the LLM never supplies one."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import Decimal

from . import vi
from .calc import CalcResult
from .num import Quantity

_NUM = re.compile(r"\d+(?:[.,]\d+)*")
_PLACEHOLDER = re.compile(r"\{(C\d+)\}")
_PAGE_REF = re.compile(r"\[\s*tr\.?[^\]]*\]|\btrang\s+\d+|\bpage\s+\d+", re.I)
CITE_RE = re.compile(r"\[tr\. (\d+(?:, \d+)*)\]")
# magnitude printed right after a number ('1.192 | Nghìn tỷ đồng', '18 triệu', '40,4%') or before it in a table cell
# ('31/12/2025 Triệu đồng: 1.192.344.137')
_SCALE_AFTER = re.compile(r"\s*(?:\|\s*)?((?:nghìn|ngàn)[\s|]+tỷ|tỷ|tỉ|triệu|nghìn|ngàn|điểm\s+phần\s+trăm|điểm\s*%|%|phần trăm)(?![^\W\d_])", re.I)  # KPI tiles split 'Nghìn | tỷ đồng'
_SCALE_BEFORE = re.compile(r"(nghìn tỷ|ngàn tỷ|tỷ|tỉ|triệu|nghìn|ngàn) (?:đồng|USD|VND)\s*:\s*$", re.I)
_YEAR = re.compile(r"^(19|20)\d{2}$")


def _norm_scale(w: str) -> str:
    w = re.sub(r"[\s|]+", " ", w.lower())
    # percentage points are printed as '%' in the report ('1,9% (▼) so với 32,7%'): same magnitude as '1,9 điểm phần trăm'
    return {"ngàn tỷ": "nghìn tỷ", "tỉ": "tỷ", "ngàn": "nghìn", "phần trăm": "%", "điểm phần trăm": "%", "điểm %": "%", "điểm%": "%"}.get(w, w)


def _scale_at(text: str, start: int, end: int) -> str | None:
    m = _SCALE_AFTER.match(text, end)
    if m: return _norm_scale(m.group(1))
    m = _SCALE_BEFORE.search(text[max(0, start - 40):start])
    return _norm_scale(m.group(1)) if m else None


_MULT = {"nghìn tỷ": 10**12, "tỷ": 10**9, "triệu": 10**6, "nghìn": 10**3, None: 1}


def figures(text: str, keep: str | None = None, small: bool = False) -> list[tuple[str, Decimal, Decimal]]:
    """Figures a claim states, as (kind, value, rounding half-unit) in base units: '1.192 nghìn tỷ' and '1.192.344 tỷ'
    both land near 1.192e15, '53,4 nghìn tỷ' carries a half-unit of 0.05e12. Years and small integers are left out
    (small=True keeps a small integer that carries a magnitude: '1 triệu')."""
    text = vi.nfc(text)
    out = []
    for m in _NUM.finditer(text):
        t = m.group(0)
        if (_trivial_at(text, m) if small else _trivial(t)) or _YEAR.match(t): continue
        c = vi.canon_num(t).rstrip("%")
        if keep is not None and c != keep: continue
        try: v = Decimal(c)
        except Exception: continue
        s = _scale_at(text, m.start(), m.end())
        kind, mult = ("%", 1) if s == "%" else ("n", _MULT.get(s, 1))
        dec = len(c.split(".")[1]) if "." in c else 0
        out.append((kind, v * mult, Decimal(mult) / (2 * 10 ** dec)))
    return out


def duplicate_claims(texts: list[str]) -> set[int]:
    """Indices of claims that only restate figures already given by an earlier claim (the same number from another
    page, or the same figure at another rounding or unit). A claim with any new figure, or none at all, is kept."""
    seen: list[tuple[str, Decimal, Decimal]] = []
    dup = set()
    for i, t in enumerate(texts):
        fs = figures(t)
        same = lambda f: any(f[0] == g[0] and abs(f[1] - g[1]) <= max(f[2], g[2]) for g in seen)
        if fs and all(same(f) for f in fs): dup.add(i)
        else: seen += fs
    return dup


def magnitude_problems(claim_text: str, cited_text: str) -> list[str]:
    """A figure copied with the wrong magnitude ('1.192 tỷ' for '1.192 nghìn tỷ') or with its magnitude dropped is
    wrong by a factor of a thousand while every digit still matches. Flag it when the cited evidence prints the same
    number with a different magnitude everywhere it occurs."""
    claim_text, cited_text = vi.nfc(claim_text), vi.nfc(cited_text)
    occ: dict[str, list[str | None]] = {}
    for m in _NUM.finditer(cited_text):
        occ.setdefault(vi.canon_num(m.group(0)).rstrip("%"), []).append(_scale_at(cited_text, m.start(), m.end()))
    out = []
    for m in _NUM.finditer(claim_text):
        t = m.group(0)
        if _trivial(t) or _YEAR.match(t): continue
        ev = occ.get(vi.canon_num(t).rstrip("%"))
        explicit = {s for s in (ev or []) if s}
        if not explicit: continue
        cs = _scale_at(claim_text, m.start(), m.end())
        if cs and cs not in explicit:
            out.append(f"magnitude of {t} differs from the evidence: '{cs}' vs '{'/'.join(sorted(explicit))}'")
        elif not cs and None not in ev:
            out.append(f"magnitude of {t} dropped (the evidence prints '{'/'.join(sorted(explicit))}')")
    return out


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
    """Numbers in canonical form only, '%' dropped: '14,6%' -> '14.6', '1.192.344.137' -> '1192344137'. Raw forms are
    not kept: the raw '179.501' (one hundred seventy-nine thousand) equals the canonical form of '179,501' (179.5)."""
    return {vi.canon_num(m.group(0)).rstrip("%") for m in _NUM.finditer(vi.nfc(text))}


def _trivial(t: str) -> bool:
    return t.isdigit() and int(t) <= 12     # "3 yếu tố", "tháng 12", list counts


def _trivial_at(text: str, m) -> bool:
    """A small integer is exempt only when no magnitude, '%' or currency follows it: '3 yếu tố' yes, '5 tỷ đồng' no."""
    return _trivial(m.group(0)) and not _scale_at(text, m.start(), m.end()) and \
        not re.match(r"\s*(?:đồng|USD|VND)", text[m.end():], re.I)


_QDATE = re.compile(r"\b\d{1,2}/\d{1,2}/(?:19|20)\d{2}\b|\b(?:19|20)\d{2}\b")


def question_numbers(question: str) -> set[str]:
    """Only the years and dates of a question may appear in a claim without evidence ('Năm 2025, ...'); any other number
    in the question ('Tài sản 999 tỷ đúng không?') is a premise to check, not evidence."""
    return {p for m in _QDATE.finditer(vi.nfc(question)) for p in re.split(r"/", m.group(0))}


def _negative_at(text: str, m) -> bool:
    """Accounting '(100)' closed right after the number, or '-100'."""
    before, after = text[:m.start()].rstrip(), text[m.end():]
    return (before.endswith("(") and after.startswith(")")) or bool(re.search(r"(?<![\d\w%])-\s?$", text[max(0, m.start() - 2):m.start()]))


def _signs(text: str) -> dict[str, set[bool]]:
    """canonical number -> {True if printed negative, False if positive} over its occurrences."""
    out: dict[str, set[bool]] = {}
    for m in _NUM.finditer(text):
        out.setdefault(vi.canon_num(m.group(0)).rstrip("%"), set()).add(_negative_at(text, m))
    return out


# words that state a printed-negative figure as a positive amount ('chi phí 17.000' for '(17.000)'), only when they stand
# just before the figure: 'lợi nhuận 100 triệu đồng sau chi phí' is a positive amount
_NEG_LEAD = re.compile(r"(?:lỗ|giảm|âm|chi phí|chi trả|dự phòng|trích lập|khấu hao|thâm hụt)"
                       r"(?:\s+(?:ròng|thuần|là|của|khoảng|đạt|ghi nhận|tương ứng))*\s*[:(]?\s*$", re.I)


def _neg_word_before(text: str, m) -> bool:
    return bool(_NEG_LEAD.search(text[max(0, m.start() - 40):m.start()]))


def sign_problems(claim_text: str, cited_text: str) -> list[str]:
    """Signs must agree both ways. A figure written negative ('-100', or '(100)' closed right after the number) must be
    printed negative in the evidence; a parenthetical such as '(64.561 triệu đồng)' is not a sign. A figure the evidence
    prints only negative may be written positive only with a word that carries the sign (lỗ, giảm, chi phí, dự phòng ...)."""
    claim_text, cited_text = vi.nfc(claim_text), vi.nfc(cited_text)
    ev = _signs(cited_text)
    out = []
    for m in _NUM.finditer(claim_text):
        t = m.group(0)
        if _trivial(t) or _YEAR.match(t): continue
        signs = ev.get(vi.canon_num(t).rstrip("%"))
        if not signs: continue
        if _negative_at(claim_text, m):
            if True not in signs: out.append(f"{t} is negative but the evidence prints it positive")
        elif signs == {True} and not _neg_word_before(claim_text, m):
            out.append(f"{t} is positive but the evidence prints it negative")
    return out


def _currency_after(text: str, m) -> str | None:
    tail = text[m.end():m.end() + 30]
    for c, rx in _CURRENCY.items():
        if re.match(rf"[\s|]*(?:nghìn tỷ|ngàn tỷ|tỷ|tỉ|triệu|nghìn|ngàn)?[\s|]*{rx}", tail, re.I):
            return c
    return None


def currency_problems(claim_text: str, cited_text: str) -> list[str]:
    """A figure printed in one currency is not the same figure in another: '100 triệu USD' is not '100 triệu đồng'.
    Checked only where both the claim and the evidence print a currency next to the figure."""
    claim_text, cited_text = vi.nfc(claim_text), vi.nfc(cited_text)
    printed: dict[str, set[str]] = {}
    for m in _NUM.finditer(cited_text):
        c = _currency_after(cited_text, m)
        if c: printed.setdefault(vi.canon_num(m.group(0)).rstrip("%"), set()).add(c)
    out = []
    for m in _NUM.finditer(claim_text):
        t = m.group(0)
        if _trivial(t) or _YEAR.match(t): continue
        c = _currency_after(claim_text, m)
        ev = printed.get(vi.canon_num(t).rstrip("%"))
        if c and ev and c not in ev:
            out.append(f"{t} is written in {c} but the evidence prints it in {'/'.join(sorted(ev))}")
    return out


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


_SCALE_OF = {10**12: "nghìn tỷ", 10**9: "tỷ", 10**6: "triệu", 10**3: "nghìn", 1: None}
_CURRENCY = {"VND": r"(?:đồng|VND)(?![^\W\d_])", "USD": r"(?:USD|đô la)(?![^\W\d_])"}


_YR = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")


def _year_of(text: str, m) -> set[str]:
    """The year a number belongs to: the last year printed before it in the same segment (a table cell follows its column
    header, 'Năm 2025 Triệu đồng: 3.503; Năm 2024 ...'; a tile its caption), else the first one after it when no other
    figure comes in between; 'năm trước' / 'cùng kỳ' right after the number means the year before. Empty if no year."""
    a = max(text.rfind(". ", 0, m.start()), text.rfind(";", 0, m.start())) + 1
    ends = [i for i in (text.find(". ", m.end()), text.find(";", m.end())) if i >= 0]
    seg_end = min(ends) if ends else len(text)
    before = [y.group(0) for y in _YR.finditer(text, a, m.start())]
    nxt = next(_YR.finditer(text, m.end(), seg_end), None)
    # a year after the number is its own only if no other figure comes between ('30,8% | 1,9% so với 32,7% trong năm
    # 2024': 2024 belongs to 32,7%, not to 30,8%)
    after = nxt.group(0) if nxt and not any(not _YR.fullmatch(x.group(0)) for x in _NUM.finditer(text, m.end(), nxt.start())) else None
    y = before[-1] if before else after
    if y is None: return set()
    out = {y}
    if re.match(r"[^.;|]{0,40}?(?:năm trước|năm ngoái|cùng kỳ)", text[m.end():], re.I):
        out.add(str(int(y) - 1))           # '12.705 người, tăng so với 11.848 người của năm trước'
    return out


def literal_quantities(plan, evs: list[Ev], question: str = "") -> tuple[dict[str, Quantity], list[str]]:
    """Quantities declared by the plan from evidence text. The model declares number, unit and period; each is checked
    against the cited block: the number occurs there; its magnitude is the one printed next to it (or, when none is
    printed next to it, a magnitude word occurs in the block); its year occurs in the block or in the question."""
    from .num import NumError, literal
    by = {e.eid: e for e in evs}
    out, probs = {}, []
    for x in getattr(plan, "literals", []):
        e = by.get(x.evidence)
        if e is None:
            probs.append(f"{x.lid}: unknown evidence id {x.evidence}"); continue
        n = vi.nfc(x.number).strip()
        text = vi.nfc(e.text)
        digits = re.sub(r"^[(\-\s]+|[)\s]+$", "", n)               # '(100)' and '-100' are looked up as '100'
        occ = [m for m in _NUM.finditer(text) if vi.canon_num(m.group(0)).rstrip("%") == vi.canon_num(digits).rstrip("%")]
        if not occ:
            probs.append(f"{x.lid}: number {x.number} not found in {x.evidence}"); continue
        try:
            q = literal(x.lid, n, x.unit, x.period, x.label, e.chunk_id, e.page)
        except NumError as err:
            probs.append(f"{x.lid}: {err}"); continue
        signs = {_negative_at(text, m) for m in occ}
        if signs == {True} and q.value > 0 or signs == {False} and q.value < 0:
            probs.append(f"{x.lid}: {x.number} has the opposite sign of the one printed in {x.evidence}"); continue
        if q.dim in _CURRENCY:
            near = {c for m in occ for c, rx in _CURRENCY.items() if re.match(rf"[\s|]*(?:nghìn tỷ|ngàn tỷ|tỷ|tỉ|triệu|nghìn|ngàn)?[\s|]*{rx}", text[m.end():], re.I)}
            if near and q.dim not in near or not near and not re.search(_CURRENCY[q.dim], text, re.I):
                probs.append(f"{x.lid}: currency of '{x.unit}' is not the one printed in {x.evidence}"); continue
        printed = {_scale_at(text, m.start(), m.end()) for m in occ} - {None}
        want = "%" if q.dim == "%" else _SCALE_OF.get(q.scale)
        if printed and want not in printed:
            probs.append(f"{x.lid}: unit '{x.unit}' but {x.evidence} prints {x.number} as '{'/'.join(sorted(printed))}'"); continue
        if not printed and want and not re.search(rf"(?<![^\W\d_]){re.escape(want)}(?![^\W\d_])", text, re.I):
            probs.append(f"{x.lid}: unit '{x.unit}' does not occur in {x.evidence}"); continue
        if q.period.year:
            got = {y for m in occ for y in _year_of(text, m)}
            if got and str(q.period.year) not in got:
                probs.append(f"{x.lid}: period {x.period} but {x.evidence} prints {x.number} under {'/'.join(sorted(got))}"); continue
            if not got and str(q.period.year) not in question:
                probs.append(f"{x.lid}: period {x.period} occurs neither in {x.evidence} nor in the question"); continue
        out[x.lid] = q
    return out, probs


def check(plan, evs: list[Ev], quantities: dict[str, Quantity], question: str = "") -> Check:
    """Deterministic validation of an LLM plan against the retrieved evidence."""
    c = Check()
    lits, lprobs = literal_quantities(plan, evs, question)
    c.plan_level += lprobs
    quantities = {**quantities, **lits}
    eids = {e.eid: e for e in evs}
    cids = {x.cid for x in plan.calculations}
    q_nums = question_numbers(question)
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
        nb = vi.nfc(body)
        bad = sorted({m.group(0) for m in _NUM.finditer(nb)
                      if not _trivial_at(nb, m) and vi.canon_num(m.group(0)).rstrip("%") not in ok_nums})
        if bad:
            ps.append("numbers not found in the cited evidence: " + ", ".join(bad[:6]))
        ps += magnitude_problems(body, cited_text)
        ps += sign_problems(body, cited_text)
        ps += currency_problems(body, cited_text)
        if ps: c.per_claim[i] = ps
    return c


_EQUIV = re.compile(r"(\d[\d.,]*\s*(?:nghìn tỷ|tỷ|triệu|nghìn)?\s*(?:đồng|USD)?)[^.;\d]{0,25}?\btương (?:đương|ứng)\s+(?:với\s+)?(?:khoảng\s+|mức\s+)?(\d[\d.,]*\s*(?:nghìn tỷ|tỷ|triệu|nghìn)?\s*(?:đồng|USD)?)", re.I)


def equivalence_problems(claim_text: str) -> list[str]:
    """'X, tương đương Y' restates X in another unit, so both must be the same amount within rounding:
    '834.241.663 triệu đồng, tương đương khoảng 776,5 nghìn tỷ đồng' puts a second, different figure next to the first
    (dev13 n-fu-eq). Only amounts with a magnitude are compared ('tương ứng 55%' is a different quantity)."""
    out = []
    for m in _EQUIV.finditer(vi.nfc(claim_text)):
        a, b = figures(m.group(1)), figures(m.group(2))
        if not a or not b or a[0][0] != "n" or b[0][0] != "n" or a[0][2] == Decimal("0.5") and b[0][2] == Decimal("0.5"):
            continue
        (_, va, ha), (_, vb, hb) = a[0], b[0]
        if abs(va - vb) > max(ha, hb):
            out.append(f"'{m.group(1).strip()}' and '{m.group(2).strip()}' are given as equal but are different amounts")
    return out


def source_ids(claim, evs: list[Ev]) -> list[str]:
    """The blocks a sentence is cited to: the first block the model lists for it, plus the blocks that print the most of
    its figures, so the analyst gets the page the figure was copied from rather than every page that repeats it. A
    sentence without figures keeps all its ids. The full list stays in the plan and the trace."""
    by = {e.eid: e for e in evs}
    ids = [e for e in claim.evidence if e in by]
    figs = {t for t in num_tokens(render_claim(claim, {})) if not _trivial(t)}
    if not ids or not figs: return ids
    score = {e: len(figs & num_tokens(by[e].text)) for e in ids}
    best = max(score.values())
    return list(dict.fromkeys(ids[:1] + [e for e in ids if score[e] == best])) if best else ids


def pages_of(claim, evs: list[Ev], plan, results: dict[str, CalcResult], quantities: dict[str, Quantity]) -> list[int]:
    by = {e.eid: e for e in evs}
    pg = {by[e].page for e in source_ids(claim, evs) if by[e].page is not None}
    refs = set(_PLACEHOLDER.findall(claim.text_vi)) | set(claim.calc_refs)
    for x in plan.calculations:
        if x.cid in refs:
            pg |= {quantities[q].page for q in x.inputs if q in quantities and quantities[q].page is not None}
    return sorted(pg)


_UNITS = r"(?:nghìn tỷ đồng|tỷ đồng|triệu đồng|nghìn đồng|đồng|triệu USD|USD|điểm phần trăm|người)"


def render_claim(claim, results: dict[str, CalcResult]) -> str:
    """Put calculator results into the sentence. The result text carries its own sign and unit, so the words the model
    wrote around the placeholder are made consistent with it: 'giảm {C1}' with -39,3% -> 'giảm 39,3%', 'tăng {C1}' with a
    negative result -> 'giảm ...', '{C1}%' -> one '%', '{C1} triệu đồng' -> one unit."""
    def put(m):
        r = results.get(m.group(2))
        if r is None: return m.group(0)
        verb, txt = m.group(1) or "", r.text
        if txt.startswith("-") and verb:
            txt = txt[1:]
            if verb.strip().lower() == "tăng": verb = verb.replace("tăng", "giảm").replace("Tăng", "Giảm")
        elif verb.strip().lower() == "giảm" and r.value > 0 and r.op in ("growth_pct", "diff", "cagr"):
            verb = verb.replace("giảm", "tăng").replace("Giảm", "Tăng")
        return verb + txt
    s = re.sub(r"((?:tăng|giảm|Tăng|Giảm)\s+(?:thêm\s+|khoảng\s+)?)?\{(C\d+)\}", put, claim.text_vi)
    s = re.sub(r"%\s*%", "%", s)
    s = re.sub(rf"({_UNITS})(\s+\1)+", r"\1", s)
    # an amount restated as itself ('27.538 tỷ đồng, tương đương 27.538 tỷ đồng') and remarks on where the report puts
    # a figure ('... được báo cáo tóm tắt ở phần điểm nhấn') add nothing the citation does not already say
    s = re.sub(r"(\d[\d.,]*\s*(?:nghìn tỷ|tỷ|triệu|nghìn)?\s*(?:đồng|USD)?),\s*tương đương\s+\1(?=[\s,.;]|$)", r"\1", s)
    s = re.sub(r",?\s*(?:và\s+)?(?:được\s+)?báo cáo\s+(?:tóm tắt|ghi nhận|trình bày|nêu)\b[^.,;]*?\b(?:ở|tại|trong)\s+(?:phần|bảng|mục)\b[^.,;]*", "", s)
    if re.search(r"\b(?:tăng|giảm)\b", s, re.I):          # the drawn-arrow marker is evidence text; once a verb states
        s = re.sub(r"\s*\([▲▼]\)", "", s)                  # the direction it only repeats it ('tăng 24% (▲)')
    return s.strip()


def marker(pages: list[int]) -> str:
    return f" [tr. {', '.join(map(str, pages))}]" if pages else ""


def with_marker(sentence: str, pages: list[int]) -> str:
    """Citation inside the sentence, before its full stop, so claims read as one paragraph: '... năm trước [tr. 59].'"""
    s = sentence.strip()
    end = s[-1] if s and s[-1] in ".!?…" else "."
    body = s[:-1].rstrip() if s and s[-1] in ".!?…" else s
    return body + marker(pages) + end


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
    """Numbers in free text (e.g. the 'missing part' note of a partial answer) not present in any evidence."""
    ok = num_tokens(" ".join(e.text for e in evs)) | num_tokens(question)
    return sorted({m.group(0) for m in _NUM.finditer(vi.nfc(text))
                   if not _trivial(m.group(0)) and vi.canon_num(m.group(0)).rstrip("%") not in ok})

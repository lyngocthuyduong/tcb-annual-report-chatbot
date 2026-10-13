"""Deterministic financial calculator. Typed, Decimal, strict about units and periods. The LLM never does arithmetic."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, DivisionByZero, InvalidOperation, localcontext

from .num import Quantity, fmt, scale_for, unit_vi


class CalcError(ValueError):
    code = "calc"


class UnitMismatch(CalcError): code = "unit"
class PeriodMismatch(CalcError): code = "period"
class ZeroDenominator(CalcError): code = "zero"
class InvalidInput(CalcError): code = "input"


@dataclass(frozen=True)
class CalcResult:
    op: str
    value: Decimal            # exact (up to 28 digits), in `dim` x `scale`
    dim: str
    scale: int
    digits: int               # display precision
    inputs: tuple[str, ...]   # qids
    formula: str

    @property
    def text(self) -> str:
        n = fmt(self.value, self.digits)
        return n + "%" if self.dim == "%" else f"{n} {unit_vi(self.dim, self.scale)}".strip()


def _same_dim(*qs: Quantity) -> None:
    if len({q.dim for q in qs}) != 1:
        raise UnitMismatch("khác loại đơn vị: " + ", ".join(f"{q.qid}={q.dim}" for q in qs))


def _same_kind(*qs: Quantity) -> None:
    """Table quantities: a balance at a date and a flow over a year never mix. Literals copied from prose (L*) carry a
    period written by the model ('2025' or '31/12/2025' for the same year-end headcount), so for them only the year counts."""
    ks = {q.period.kind for q in qs}
    lit = any(q.qid.startswith("L") for q in qs)
    if "none" in ks or (len(ks) != 1 and not (lit and ks <= {"date", "year"})):
        raise PeriodMismatch("kỳ không xác định hoặc khác loại: " + ", ".join(f"{q.qid}={q.period.kind}" for q in qs))


def _same_period(*qs: Quantity) -> None:
    _same_kind(*qs)
    if len({q.period.year for q in qs}) != 1:
        raise PeriodMismatch("khác kỳ: " + ", ".join(f"{q.qid}={q.period.label}" for q in qs))


def _need(q: Quantity) -> Quantity:
    if not isinstance(q, Quantity):
        raise InvalidInput(f"not a Quantity: {q!r}")
    return q


def growth_pct(q_curr: Quantity, q_prev: Quantity) -> CalcResult:
    """(curr - prev) / |prev| * 100. Same unit dimension, same period kind, curr later than prev, prev != 0."""
    _need(q_curr); _need(q_prev)
    _same_dim(q_curr, q_prev); _same_kind(q_curr, q_prev)
    if not (q_curr.period.year and q_prev.period.year and q_curr.period.year > q_prev.period.year):
        raise PeriodMismatch(f"{q_curr.qid} phải thuộc kỳ sau {q_prev.qid}")
    if q_prev.base == 0:
        raise ZeroDenominator(f"{q_prev.qid} = 0")
    v = (q_curr.base - q_prev.base) / abs(q_prev.base) * 100
    return CalcResult("growth_pct", v, "%", 1, 1, (q_curr.qid, q_prev.qid), f"({q_curr.qid}-{q_prev.qid})/|{q_prev.qid}|*100")


def diff(q_a: Quantity, q_b: Quantity) -> CalcResult:
    """a - b. Same unit dimension and period kind. % - % -> percentage points."""
    _need(q_a); _need(q_b)
    _same_dim(q_a, q_b); _same_kind(q_a, q_b)
    if q_a.dim == "%":
        return CalcResult("diff", q_a.value - q_b.value, "pp", 1, 2, (q_a.qid, q_b.qid), f"{q_a.qid}-{q_b.qid}")
    whole = q_a.scale == q_b.scale and all(q.value == q.value.to_integral_value() for q in (q_a, q_b))
    return CalcResult("diff", (q_a.base - q_b.base) / q_a.scale, q_a.dim, q_a.scale, 0 if whole else 1, (q_a.qid, q_b.qid),
                      f"{q_a.qid}-{q_b.qid}")


def ratio(q_num: Quantity, q_den: Quantity) -> CalcResult:
    """num / den as a plain multiple (lần). Same unit dimension; den != 0."""
    _need(q_num); _need(q_den)
    _same_dim(q_num, q_den)
    if q_den.base == 0:
        raise ZeroDenominator(f"{q_den.qid} = 0")
    return CalcResult("ratio", q_num.base / q_den.base, "x", 1, 2, (q_num.qid, q_den.qid), f"{q_num.qid}/{q_den.qid}")


def share_pct(q_part: Quantity, q_total: Quantity) -> CalcResult:
    """part / total * 100. Same unit dimension and same period; total != 0."""
    _need(q_part); _need(q_total)
    _same_dim(q_part, q_total); _same_period(q_part, q_total)
    if q_total.base == 0:
        raise ZeroDenominator(f"{q_total.qid} = 0")
    return CalcResult("share_pct", q_part.base / q_total.base * 100, "%", 1, 1, (q_part.qid, q_total.qid),
                      f"{q_part.qid}/{q_total.qid}*100")


def cagr(q_start: Quantity, q_end: Quantity) -> CalcResult:
    """((end/start)^(1/years) - 1) * 100. Same unit dimension and kind, years >= 1, start > 0, end >= 0."""
    _need(q_start); _need(q_end)
    _same_dim(q_start, q_end); _same_kind(q_start, q_end)
    years = (q_end.period.year or 0) - (q_start.period.year or 0)
    if years < 1:
        raise PeriodMismatch(f"{q_end.qid} phải sau {q_start.qid} ít nhất 1 năm")
    if q_start.base <= 0 or q_end.base < 0:
        raise InvalidInput("CAGR cần giá trị đầu > 0 và giá trị cuối >= 0")
    with localcontext() as c:
        c.prec = 28
        try:
            v = ((q_end.base / q_start.base) ** (Decimal(1) / Decimal(years)) - 1) * 100
        except (InvalidOperation, DivisionByZero) as e:
            raise InvalidInput(str(e))
    return CalcResult("cagr", v, "%", 1, 1, (q_start.qid, q_end.qid), f"(({q_end.qid}/{q_start.qid})^(1/{years})-1)*100")


def sum_(qs: list[Quantity]) -> CalcResult:
    """Sum of >= 1 quantities with the same unit dimension and the same period. Result in the first item's scale."""
    if not qs:
        raise InvalidInput("danh sách rỗng")
    for q in qs: _need(q)
    _same_dim(*qs); _same_period(*qs)
    first = qs[0]
    total = sum((q.base for q in qs), Decimal(0)) / first.scale
    return CalcResult("sum", total, first.dim, first.scale, 0 if total == total.to_integral_value() else 1,
                      tuple(q.qid for q in qs), "+".join(q.qid for q in qs))


def convert(q: Quantity, to: str) -> CalcResult:
    """Rescale a money quantity (VND or USD only): e.g. triệu đồng -> tỷ đồng."""
    _need(q)
    try:
        dim, sc = scale_for(to)
    except ValueError as e:
        raise InvalidInput(str(e))
    if dim != q.dim or dim not in ("VND", "USD"):
        raise UnitMismatch(f"không đổi {q.dim} -> {dim}")
    return CalcResult("convert", q.base / sc, dim, sc, 3, (q.qid,), f"{q.qid} -> {to}")


OPS = ("growth_pct", "diff", "ratio", "share_pct", "cagr", "sum", "convert")


def run(op: str, qs: list[Quantity], to: str | None = None) -> CalcResult:
    """Dispatch a plan step. Wrong arity / unknown op -> InvalidInput."""
    arity = {"growth_pct": 2, "diff": 2, "ratio": 2, "share_pct": 2, "cagr": 2, "convert": 1}
    if op not in OPS:
        raise InvalidInput(f"unknown op {op!r}")
    if op != "sum" and len(qs) != arity[op]:
        raise InvalidInput(f"{op} needs {arity[op]} inputs, got {len(qs)}")
    if op == "sum": return sum_(qs)
    if op == "convert":
        if not to: raise InvalidInput("convert needs 'to'")
        return convert(qs[0], to)
    a, b = qs
    if op in ("growth_pct", "cagr") and a.period.year and b.period.year and a.period.year != b.period.year:
        later_first = op == "growth_pct"                      # growth_pct(current, previous); cagr(start, end)
        if (a.period.year > b.period.year) != later_first: a, b = b, a
    return {"growth_pct": growth_pct, "diff": diff, "ratio": ratio, "share_pct": share_pct, "cagr": cagr}[op](a, b)

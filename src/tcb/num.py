"""Quantities: Vietnamese number/unit/period parsing and display. No LLM. Decimal only."""
from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal

from . import vi

_VN = re.compile(r"^\(?-?\d{1,3}(?:\.\d{3})+(?:,\d+)?\)?$|^\(?-?\d+(?:,\d+)?\)?$")
_DATE = re.compile(r"(\d{1,2})/(\d{1,2})/(\d{4})")
_YEAR = re.compile(r"(?<!\d)(19|20)\d{2}(?!\d)")


class NumError(ValueError):
    pass


def parse_vn(s: str) -> Decimal | None:
    """'1.192.344.137' -> 1192344137; '14,6' -> 14.6; '(181.150)' -> -181150; '-' -> None (nil cell)."""
    t = vi.nfc(s).strip().rstrip("%").strip()
    if t in ("", "-", "–", "—"):
        return None
    if not _VN.match(t):
        raise NumError(f"not a Vietnamese number: {s!r}")
    neg = t.startswith("(") or t.startswith("-")
    t = t.strip("()-").replace(".", "").replace(",", ".")
    d = Decimal(t)
    return -d if neg else d


# (folded unit text, dimension, scale to the base unit of that dimension)
_UNITS: list[tuple[str, str, int]] = [
    ("nghin ty dong", "VND", 10**12), ("nghin ty vnd", "VND", 10**12),
    ("ty dong", "VND", 10**9), ("ty vnd", "VND", 10**9),
    ("trieu dong", "VND", 10**6), ("trieu vnd", "VND", 10**6),
    ("nghin dong", "VND", 10**3), ("nghin vnd", "VND", 10**3),
    ("dong", "VND", 1), ("vnd", "VND", 1),
    ("ty usd", "USD", 10**9), ("trieu usd", "USD", 10**6), ("nghin usd", "USD", 10**3), ("usd", "USD", 1),
    ("diem phan tram", "pp", 1), ("pp", "pp", 1),
    ("%", "%", 1), ("phan tram", "%", 1),
    ("lan", "x", 1),
    ("trieu khach hang", "count", 10**6), ("nghin khach hang", "count", 10**3), ("khach hang", "count", 1),
    ("trieu", "count", 10**6), ("nghin", "count", 10**3),
]


def parse_unit(s: str) -> tuple[str, int] | None:
    f = vi.fold(vi.nfc(s).lower()).strip(" .,:;()")
    for u, dim, sc in _UNITS:
        if f == u or f.endswith(" " + u) or f.endswith("(" + u + ")") or (u == "%" and f.endswith("%")):
            return dim, sc
    return None


@dataclass(frozen=True)
class Period:
    kind: str                 # date | year | none
    year: int | None = None
    label: str = ""


NO_PERIOD = Period("none")


def parse_period(s: str) -> Period:
    m = _DATE.search(s)
    if m:
        return Period("date", int(m.group(3)), m.group(0))
    m = _YEAR.search(s)
    if m:
        return Period("year", int(m.group(0)), m.group(0))
    return NO_PERIOD


@dataclass(frozen=True)
class Quantity:
    qid: str
    label: str
    value: Decimal
    dim: str                  # VND | USD | % | pp | x | count
    scale: int                # multiplier to the dimension's base unit (VND: 1e6 = triệu đồng)
    period: Period
    chunk_id: str
    page: int | None          # printed page, from node metadata
    raw: str = ""

    @property
    def base(self) -> Decimal:
        return self.value * self.scale

    def show(self) -> str:
        return f"{fmt(self.value)} {unit_vi(self.dim, self.scale)}".strip()


_SCALE_NAME = {("VND", 1): "đồng", ("VND", 10**3): "nghìn đồng", ("VND", 10**6): "triệu đồng",
               ("VND", 10**9): "tỷ đồng", ("VND", 10**12): "nghìn tỷ đồng",
               ("USD", 1): "USD", ("USD", 10**3): "nghìn USD", ("USD", 10**6): "triệu USD", ("USD", 10**9): "tỷ USD",
               ("%", 1): "%", ("pp", 1): "điểm phần trăm", ("x", 1): "lần",
               ("count", 1): "", ("count", 10**3): "nghìn", ("count", 10**6): "triệu"}


def unit_vi(dim: str, scale: int) -> str:
    return _SCALE_NAME.get((dim, scale), f"{dim}×{scale}")


def scale_for(unit_vi_text: str) -> tuple[str, int]:
    p = parse_unit(unit_vi_text)
    if not p:
        raise NumError(f"unknown unit: {unit_vi_text!r}")
    return p


def fmt(v: Decimal, digits: int | None = None) -> str:
    """Vietnamese display: 1.192.344.137 / 14,6. digits=None keeps the natural scale (<=6 dp, trailing zeros trimmed)."""
    if digits is None:
        q = v.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)
        s = format(q, "f").rstrip("0").rstrip(".")
    else:
        s = format(v.quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP), "f")
    ip, _, fp = s.partition(".")
    neg = ip.startswith("-")
    ip = ip.lstrip("-")
    ip = f"{int(ip):,}".replace(",", ".")
    return ("-" if neg else "") + ip + ("," + fp if fp else "")


def from_cell(qid: str, label: str, header: str, cell: str, unit_hint: str | None, chunk_id: str,
              page: int | None) -> Quantity | None:
    """One table cell -> Quantity. Header carries the date and (usually) the unit. None for nil/unparseable cells."""
    try:
        v = parse_vn(cell)
    except NumError:
        return None
    if v is None:
        return None
    u = parse_unit(header) or (parse_unit(unit_hint) if unit_hint else None)
    if u is None:
        return None
    return Quantity(qid, label, v, u[0], u[1], parse_period(header), chunk_id, page, cell)


def from_node(n: dict, next_id) -> list[Quantity]:
    """Quantities of one index node (dict). Only reliably-parsed ('high') financial-statement tables yield any."""
    if n.get("content_type") != "table" or n.get("quality") != "high" or not n.get("rows"):
        return []
    out = []
    for r in n["rows"]:
        for hdr, cell in r["cells"].items():
            if cell is None:
                continue
            q = from_cell(next_id(), r["label"], hdr, cell, n.get("unit_header"), n["chunk_id"], n.get("printed_page_number"))
            if q:
                out.append(q)
    return out

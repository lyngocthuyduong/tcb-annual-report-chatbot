"""PDF layout: words+bbox via poppler, spread split, printed-page map, furniture strip."""
from __future__ import annotations

import html
import re
import subprocess
from dataclasses import dataclass, field

from . import conf

STRIP_Y = 40.0  # top furniture strip (page number + running header) lies above this y


@dataclass
class Word:
    x0: float
    y0: float
    x1: float
    y1: float
    text: str
    flow: int = 0
    block: int = 0
    line: int = 0

    @property
    def xc(self) -> float:
        return (self.x0 + self.x1) / 2


@dataclass
class Line:
    words: list[Word]

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)

    @property
    def x0(self) -> float: return min(w.x0 for w in self.words)
    @property
    def x1(self) -> float: return max(w.x1 for w in self.words)
    @property
    def y0(self) -> float: return min(w.y0 for w in self.words)
    @property
    def y1(self) -> float: return max(w.y1 for w in self.words)


@dataclass
class Block:
    lines: list[Line]

    @property
    def text(self) -> str:
        return "\n".join(l.text for l in self.lines)

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return (min(l.x0 for l in self.lines), min(l.y0 for l in self.lines),
                max(l.x1 for l in self.lines), max(l.y1 for l in self.lines))


@dataclass
class Half:
    pdf_page_index: int          # 1-based
    side: str                    # "left" | "right" | "single"
    width: float
    height: float
    x_off: float                 # x offset of this half in the PDF page
    blocks: list[Block] = field(default_factory=list)   # body (furniture removed)
    furniture: list[str] = field(default_factory=list)  # strip text (running header)
    printed_page: int | None = None
    page_source: str = "footer"  # footer | inferred
    straddle: int = 0            # words crossing the gutter


# ---------------------------------------------------------------- reading
def run_poppler(pdf: str) -> str:
    return subprocess.run([conf.poppler("pdftotext"), "-bbox-layout", pdf, "-"], check=True,
                          capture_output=True, text=True, encoding="utf-8").stdout


_TOK = re.compile(r'<page width="([\d.]+)" height="([\d.]+)">|<(flow)>|<(block) |<(line) |'
                  r'<word xMin="([\d.]+)" yMin="([\d.]+)" xMax="([\d.]+)" yMax="([\d.]+)">(.*?)</word>')


def read_words(xml_text: str) -> list[tuple[float, float, list[Word]]]:
    """-> [(width, height, words)] per PDF page, words tagged with poppler flow/block/line ids.
    Regex parse: poppler's output is HTML-ish, not strict XML."""
    pages: list[tuple[float, float, list[Word]]] = []
    fi = bi = li = -1
    cur: list[Word] | None = None
    for m in _TOK.finditer(xml_text):
        if m.group(1):
            cur = []
            pages.append((float(m.group(1)), float(m.group(2)), cur))
            fi = bi = li = -1
        elif m.group(3): fi += 1; bi = li = -1
        elif m.group(4): bi += 1; li = -1
        elif m.group(5): li += 1
        elif cur is not None:
            txt = html.unescape(m.group(10)).strip()
            cur.append(Word(float(m.group(6)), float(m.group(7)), float(m.group(8)), float(m.group(9)),
                            txt, fi, bi, li))
    return pages


# ---------------------------------------------------------------- spread split
def _assign_side(w: Word, gutter: float) -> str:
    if w.x1 <= gutter: return "left"
    if w.x0 >= gutter: return "right"
    # straddling word: side with larger overlap
    return "left" if (gutter - w.x0) >= (w.x1 - gutter) else "right"


def _group(words: list[Word]) -> list[Block]:
    """Rebuild blocks/lines from poppler ids, keeping poppler's reading order."""
    order: dict[tuple, list[Word]] = {}
    for w in words:
        order.setdefault((w.flow, w.block, w.line), []).append(w)
    blocks: dict[tuple, list[Line]] = {}
    for (f, b, _l), ws in order.items():
        blocks.setdefault((f, b), []).append(Line(sorted(ws, key=lambda w: w.x0)))
    out = []
    for (_f, _b), lines in blocks.items():
        lines.sort(key=lambda l: (round(l.y0, 0), l.x0))
        out.append(Block(lines))
    return out


def split_page(idx: int, width: float, height: float, words: list[Word]) -> list[Half]:
    words = [w for w in words if w.text]
    if width < 700:  # single A4 (cover)
        h = Half(idx, "single", width, height, 0.0)
        h.blocks = _group(words)
        return [h]
    gutter = width / 2
    halves = {"left": Half(idx, "left", gutter, height, 0.0),
              "right": Half(idx, "right", gutter, height, gutter)}
    buckets: dict[str, list[Word]] = {"left": [], "right": []}
    for w in words:
        side = _assign_side(w, gutter)
        if w.x0 < gutter < w.x1: halves[side].straddle += 1
        buckets[side].append(w)
    for side, ws in buckets.items():
        furn = [w for w in ws if w.y1 <= STRIP_Y]
        body = [w for w in ws if w.y1 > STRIP_Y]
        halves[side].furniture = [" ".join(w.text for w in sorted(furn, key=lambda w: w.x0))]
        halves[side].blocks = _group(body)
        halves[side]._strip = furn  # type: ignore[attr-defined]
    return [halves["left"], halves["right"]]


# ---------------------------------------------------------------- page map
def linear_map(idx: int) -> tuple[int, int]:
    """PDF page k (k>=2) holds printed 2(k-1) and 2k-1. k=1 is the cover (printed 1)."""
    return 2 * (idx - 1), 2 * idx - 1


def footer_number(h: Half) -> int | None:
    """Printed number = the digits-only word in the top strip at the outer margin."""
    strip = getattr(h, "_strip", [])
    nums = [w for w in strip if re.fullmatch(r"\d{1,3}", w.text)]
    if not nums: return None
    # outer margin: left half -> leftmost, right half -> rightmost
    pick = min(nums, key=lambda w: w.x0) if h.side == "left" else max(nums, key=lambda w: w.x1)
    return int(pick.text)


class PageMapError(RuntimeError):
    pass


def build_page_map(halves: list[Half]) -> dict:
    """Assign printed pages. Footer wins when present and must equal the linear map (else hard error)."""
    info = {"footer": 0, "inferred": 0, "conflicts": []}
    for h in halves:
        if h.side == "single":
            h.printed_page, h.page_source = 1, "inferred"
            info["inferred"] += 1
            continue
        lin = linear_map(h.pdf_page_index)[0 if h.side == "left" else 1]
        got = footer_number(h)
        if got is None:
            h.printed_page, h.page_source = lin, "inferred"
            info["inferred"] += 1
        elif got != lin:
            info["conflicts"].append((h.pdf_page_index, h.side, got, lin))
        else:
            h.printed_page, h.page_source = got, "footer"
            info["footer"] += 1
    if info["conflicts"]:
        raise PageMapError(f"footer vs linear map conflicts: {info['conflicts'][:10]}")
    return info


def direction_arrows(pdf: str) -> dict[int, list[tuple[float, float, float, float, str]]]:
    """Up/down triangles drawn before KPI change figures ('▲ 7% so với năm 2023'). They are vector paths, not text,
    so pdftotext drops them and a change loses its direction. -> {pdf page: [(x0, top, x1, bottom, 'up'|'down')]},
    coordinates in pdftotext's top-left system."""
    from pdfminer.high_level import extract_pages
    from pdfminer.layout import LTCurve, LTFigure

    def walk(o):
        for x in o:
            if isinstance(x, LTFigure): yield from walk(x)
            elif isinstance(x, LTCurve): yield x
    out: dict[int, list] = {}
    for i, page in enumerate(extract_pages(pdf, laparams=None), start=1):
        for c in walk(page):
            if not (2.5 <= c.x1 - c.x0 <= 7 and 1.2 <= c.y1 - c.y0 <= 4.5): continue      # glyph-sized
            pts = {(round(x, 2), round(y, 2)) for x, y in c.pts}
            if len(pts) != 3: continue                                                  # a triangle
            ys = sorted(y for _, y in pts)
            d = "up" if abs(ys[0] - ys[1]) < 0.2 else "down" if abs(ys[1] - ys[2]) < 0.2 else None
            if d: out.setdefault(i, []).append((c.x0, page.height - c.y1, c.x1, page.height - c.y0, d))
    return out


ARROW_WORD = {"up": "(▲)", "down": "(▼)"}
# Written after the figure, not as 'tăng X': the triangle marks the direction of change, while the figure beside it is
# sometimes the change ('▲ 7% so với năm 2023') and sometimes the level ('▲ 40,4%' for the CASA ratio).


def add_arrows(words: list[Word], arrows: list[tuple]) -> int:
    """Turn each triangle that sits just left of a figure on the same line into a marker '(▲)'/'(▼)' placed right
    after that figure: '93% | 7% (▲) so với năm 2023'."""
    n = 0
    for x0, y0, x1, y1, d in arrows:
        yc = (y0 + y1) / 2
        nxt = [w for w in words if abs((w.y0 + w.y1) / 2 - yc) < 3 and 0 <= w.x0 - x1 <= 8 and re.match(r"[\d(+-]", w.text)]
        if not nxt: continue
        w = min(nxt, key=lambda w: w.x0)
        words.append(Word(w.x1 + 0.01, w.y0, w.x1 + 0.02, w.y1, ARROW_WORD[d], w.flow, w.block, w.line))
        n += 1
    return n


def load(pdf: str) -> tuple[list[Half], dict]:
    pages = read_words(run_poppler(pdf))
    arrows = direction_arrows(pdf)
    halves: list[Half] = []
    n_arrows = 0
    for i, (w, h, words) in enumerate(pages, start=1):
        n_arrows += add_arrows(words, arrows.get(i, []))
        halves += split_page(i, w, h, words)
    info = build_page_map(halves)
    info["pdf_pages"] = len(pages)
    info["halves"] = len(halves)
    info["straddle_words"] = sum(h.straddle for h in halves)
    info["direction_arrows"] = n_arrows
    return halves, info

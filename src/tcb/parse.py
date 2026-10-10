"""Half-page -> typed items: narrative groups, KPI tiles, FS table rows, glossary terms, image inventory."""
from __future__ import annotations

import re
import statistics
import subprocess
from dataclasses import dataclass, field

from . import conf
from .pdf import Block, Half, Word

# ---------------------------------------------------------------- numbers
STRICT = re.compile(r"^\(?-?\d{1,3}(?:\.\d{3})+(?:,\d+)?\)?%?$|^\(?-?\d+,\d+\)?%?$")
PLAIN = re.compile(r"^\(?-?\d+\)?%?$")
NOTE_REF = re.compile(r"^\d{1,2}(?:\.\d{1,2})?$")
FORM = re.compile(r"B\d{2}[a-z]?/TCTD-HN")
UNIT_RE = re.compile(r"(Nghìn tỷ đồng|Triệu đồng|Tỷ đồng|Nghìn đồng|Triệu USD|Triệu|Đồng|%)", re.I)
DATE_RE = re.compile(r"\d{2}/\d{2}/\d{4}")
SECTION_FS = "Báo cáo tài chính"


def is_num(tok: str, strict_only: bool = False) -> bool:
    if tok == "-": return True
    if STRICT.match(tok): return True
    return (not strict_only) and bool(PLAIN.match(tok))


def section_of(halves_by_page: dict, idx: int) -> str:
    """Chapter name = right-half running header minus page number."""
    for side in ("right", "single"):
        h = halves_by_page.get((idx, side))
        if h and h.furniture:
            t = re.sub(r"\b\d{1,3}\b", "", h.furniture[0]).replace("Báo cáo thường niên 2025", "").strip()
            return t
    return ""


def half_words(h: Half) -> list[Word]:
    return [w for b in h.blocks for l in b.lines for w in l.words]


# ---------------------------------------------------------------- grouping (narrative + KPI)
@dataclass
class Group:
    blocks: list[Block]
    kind: str = "narrative"       # narrative | kpi
    order: tuple = (0, 0)

    @property
    def text(self) -> str:
        return "\n".join(b.text for b in sorted(self.blocks, key=lambda b: (round(b.bbox[1]), b.bbox[0])))

    @property
    def bbox(self):
        bb = [b.bbox for b in self.blocks]
        return (min(b[0] for b in bb), min(b[1] for b in bb), max(b[2] for b in bb), max(b[3] for b in bb))


def _link(a: Block, b: Block) -> bool:
    ax0, ay0, ax1, ay1 = a.bbox
    bx0, by0, bx1, by1 = b.bbox
    ox = min(ax1, bx1) - max(ax0, bx0)               # horizontal overlap
    oy = min(ay1, by1) - max(ay0, by0)               # vertical overlap
    dx = max(ax0, bx0) - min(ax1, bx1)               # horizontal gap
    dy = max(ay0, by0) - min(ay1, by1)               # vertical gap
    narrow = min(ax1 - ax0, bx1 - bx0)
    small = len(a.lines) <= 3 and len(b.lines) <= 3
    if ox >= 0.4 * narrow and dy <= (20 if small else 12): return True  # stacked
    if oy > 0 and dx <= 12 and dx >= 0 and oy >= 0.5 * min(ay1 - ay0, by1 - by0): return True  # side by side (value | unit)
    return False


def groups(blocks: list[Block]) -> list[Group]:
    n = len(blocks)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            if _link(blocks[i], blocks[j]):
                parent[find(i)] = find(j)
    comp: dict[int, list[int]] = {}
    for i in range(n): comp.setdefault(find(i), []).append(i)
    out = []
    for idxs in comp.values():
        bl = [blocks[i] for i in idxs]
        g = Group(bl, order=min(i for i in idxs) and (min(idxs),) or (0,))
        g.order = (min(idxs),)
        # KPI: short group holding a big numeric block
        chars = sum(len(b.text) for b in bl)
        big_num = any((b.bbox[3] - b.bbox[1]) >= 20 and re.search(r"\d", b.text) and len(b.text) <= 12 for b in bl)
        if big_num and chars <= 220: g.kind = "kpi"
        out.append(g)
    out.sort(key=lambda g: g.order)
    return out


# ---------------------------------------------------------------- FS tables
@dataclass
class Row:
    label: str
    note: str | None
    cells: list[str | None]
    indent: float
    y: float


@dataclass
class Table:
    table_id: str
    form_code: str | None
    title: str
    continuation: bool
    headers: list[str]          # one per numeric column
    units: list[str | None]
    rows: list[Row]
    y_range: tuple[float, float]
    reliable: bool = True       # headers resolved cleanly (dates / short); else narrative view is kept too


def _rows(words: list[Word], tol: float = 3.0) -> list[list[Word]]:
    ws = sorted(words, key=lambda w: (w.y0, w.x0))
    rows: list[list[Word]] = []
    for w in ws:
        if rows and abs(rows[-1][0].y0 - w.y0) <= tol:
            rows[-1].append(w)
        else:
            rows.append([w])
    return [sorted(r, key=lambda w: w.x0) for r in rows]


def _split_row(row: list[Word], width: float, x_off: float, note_x: float | None):
    """-> (label_words, note_word|None, numeric_words). Numeric cells = trailing words in cell form
    (gap >= 8pt from previous word, at the right of the half)."""
    label, nums, note = [], [], None
    prev_x1 = None
    for w in row:
        rx = w.x0 - x_off
        gap = (w.x0 - prev_x1) if prev_x1 is not None else 99
        isn = is_num(w.text) and rx >= 0.45 * width and gap >= 8
        if note_x is not None and abs((w.x0 + w.x1) / 2 - x_off - note_x) <= 16 and NOTE_REF.match(w.text) and gap >= 8:
            note = w
        elif isn:
            nums.append(w)
        else:
            if nums:  # text after numbers: treat everything as label (not a data row)
                label += nums; nums = []
            label.append(w)
        prev_x1 = w.x1
    return label, note, nums


def tables(h: Half) -> tuple[list[Table], list[tuple[float, float]]]:
    """Detect table zones in an FS half. Returns tables and their y-ranges (to exclude from narrative)."""
    words = half_words(h)
    if not words: return [], []
    rows = _rows(words)
    note_x = None
    for r in rows:  # "Thuyết minh" header gives the note column
        for i, w in enumerate(r[:-1]):
            if w.text == "Thuyết" and r[i + 1].text == "minh":
                note_x = ((w.x0 + r[i + 1].x1) / 2) - h.x_off
    parsed = []
    for r in rows:
        lab, note, nums = _split_row(r, h.width, h.x_off, note_x)
        parsed.append((r, lab, note, nums))
    data_idx = [i for i, (_, lab, note, nums) in enumerate(parsed)
                if nums and (any(is_num(w.text, True) for w in nums) or len(nums) >= 2)]
    if len(data_idx) < 3: return [], []
    # zones = runs of data rows with gaps <= 6 rows between (label-only / header rows allowed)
    zones, cur = [], [data_idx[0]]
    for i in data_idx[1:]:
        if i - cur[-1] <= 6: cur.append(i)
        else: zones.append(cur); cur = [i]
    zones.append(cur)
    out, yr = [], []
    for zi, z in enumerate(zones):
        if len(z) < 3: continue
        first, last = z[0], z[-1]
        # extend upward over header rows (dates / units / "Thuyết minh") up to 8 rows
        top = first
        while top > 0 and first - top < 8:
            txt = " ".join(w.text for w in parsed[top - 1][0])
            gapy = parsed[top][0][0].y0 - parsed[top - 1][0][0].y0
            if gapy > 40: break
            if DATE_RE.search(txt) or UNIT_RE.search(txt) or "Thuyết minh" in txt or re.search(r"\b(Năm|Quý|Cuối|Số)\b", txt) or not parsed[top - 1][3]:
                top -= 1
            else: break
        # column anchors from right edges of numeric words
        xs = sorted(w.x1 - h.x_off for i in z for w in parsed[i][3])
        anchors: list[list[float]] = []
        for x in xs:
            if anchors and x - anchors[-1][-1] <= 14: anchors[-1].append(x)
            else: anchors.append([x])
        # merge near anchors by median
        cols = [statistics.median(a) for a in anchors if len(a) >= max(2, len(z) // 8)]
        if not cols: continue
        ncol = len(cols)
        header_rows = [parsed[i][0] for i in range(top, first)]
        hdr_words = [w for r in header_rows for w in r]
        headers, units = [], []
        for ci, cx in enumerate(cols):
            lo = (cols[ci - 1] + cx) / 2 if ci else cx - 140
            hi = cx + 14
            hw = sorted([w for w in hdr_words if lo <= (w.x1 - h.x_off) <= hi and (w.x0 - h.x_off) >= lo - 30
                         and not (note_x is not None and abs((w.x0 + w.x1) / 2 - h.x_off - note_x) < 20)],
                        key=lambda w: (round(w.y0 / 4), w.x0))
            t = " ".join(w.text for w in hw).strip()
            m = UNIT_RE.search(t)
            units.append(m.group(1) if m else None)
            headers.append(t or f"cột {ci + 1}")
        rws: list[Row] = []
        pending: list[str] = []
        for i in range(first, last + 1):
            r, lab, note, nums = parsed[i]
            if not nums:
                txt = " ".join(w.text for w in lab).strip()
                if txt: pending.append(txt)
                continue
            cells: list[str | None] = [None] * ncol
            for w in nums:
                ci = min(range(ncol), key=lambda k: abs((w.x1 - h.x_off) - cols[k]))
                cells[ci] = w.text
            label = " ".join(w.text for w in lab).strip()
            if pending and not label:  # wrapped label above the numbers
                label = " ".join(pending); pending = []
            elif pending:
                label = " ".join(pending) + " " + label; pending = []
            rws.append(Row(label, note.text if note else None, cells,
                           (lab[0].x0 - h.x_off) if lab else 0.0, r[0].y0))
        if not rws: continue
        # title + form code from the top of the page (above the zone)
        ztop_y = parsed[top][0][0].y0
        head = [w for w in words if w.y0 < ztop_y - 2]
        form = None
        for w in head:
            m = FORM.search(w.text)
            if m: form = m.group(0)
        tl = _rows([w for w in head if not FORM.search(w.text)])
        title = " ".join(" ".join(w.text for w in r) for r in tl[-4:]).strip() if zi == 0 else \
            " ".join(" ".join(w.text for w in r) for r in tl[-2:]).strip()
        title = re.sub(r"\s+", " ", title)
        reliable = all(DATE_RE.search(x) for x in headers) or \
            (ncol <= 3 and all(len(x) <= 45 and not x.startswith("cột") for x in headers))
        yy = (parsed[top][0][0].y0 - 2, parsed[last][0][0].y1 + 2)
        out.append(Table(f"{form or 'T'}@{h.printed_page}.{zi}", form, title, "tiếp theo" in title.lower(),
                         headers, units, rws, yy, reliable))
        if reliable: yr.append(yy)
    return out, yr


# ---------------------------------------------------------------- glossary
def glossary(h: Half) -> list[tuple[str, str]]:
    words = half_words(h)
    if not words: return []
    # header row "Thuật ngữ | Định nghĩa" (x0 of each) defines column pairs
    hdr = [w for w in words if w.text in ("Thuật", "Định")]
    if not hdr: return []
    y_hdr = min(w.y0 for w in hdr if w.text == "Thuật") if any(w.text == "Thuật" for w in hdr) else 0
    hdr = [w for w in hdr if abs(w.y0 - y_hdr) < 3]
    term_x = sorted(w.x0 - h.x_off for w in hdr if w.text == "Thuật")
    def_x = sorted(w.x0 - h.x_off for w in hdr if w.text == "Định")
    if not term_x or len(term_x) != len(def_x): return []
    hy = max(w.y1 for w in hdr)
    out = []
    bounds = term_x[1:] + [h.width]
    for t0, d0, nxt in zip(term_x, def_x, bounds):
        col = [w for w in words if w.y0 > hy and t0 - 3 <= (w.x0 - h.x_off) < nxt - 3]
        tw = [w for w in col if (w.x0 - h.x_off) < d0 - 3]
        dw = [w for w in col if (w.x0 - h.x_off) >= d0 - 3]
        trows = _rows(tw, 2.0)
        starts = []
        for r in trows:
            y = r[0].y0
            if starts and y - starts[-1][1] <= 14: starts[-1][1] = y; starts[-1][2].append(r)
            else: starts.append([y, y, [r]])
        for k, (y0, _ylast, rs) in enumerate(starts):
            y_end = starts[k + 1][0] - 2 if k + 1 < len(starts) else 1e9
            term = " ".join(w.text for r in rs for w in r)
            defn = " ".join(w.text for r in _rows([w for w in dw if y0 - 2 <= w.y0 < y_end], 2.0) for w in r)
            if term and defn: out.append((term.strip(), defn.strip()))
    return out


# ---------------------------------------------------------------- images
def image_inventory(pdf: str) -> dict:
    """pdfimages -list: per-page image count + physical-size share. Inventory only (no extraction)."""
    txt = subprocess.run([conf.poppler("pdfimages"), "-list", pdf], capture_output=True, text=True, encoding="utf-8",
                         errors="replace", check=True).stdout.splitlines()[2:]
    per: dict[int, dict] = {}
    for ln in txt:
        p = ln.split()
        if len(p) < 15: continue
        page, typ, w, hgt, xppi, yppi = int(p[0]), p[2], int(p[3]), int(p[4]), float(p[12]), float(p[13])
        if typ == "smask": continue
        area_pt2 = (w / max(xppi, 1) * 72) * (hgt / max(yppi, 1) * 72)
        d = per.setdefault(page, {"n": 0, "max_share": 0.0})
        d["n"] += 1
        d["max_share"] = max(d["max_share"], area_pt2 / (1190.55 * 841.89))
    big = sorted(((p, round(d["max_share"], 2)) for p, d in per.items() if d["max_share"] >= 0.25), key=lambda x: x[0])
    return {"images": sum(d["n"] for d in per.values()), "pages_with_images": len(per),
            "pages_with_large_image(>=25% area)": len(big), "large": big[:60]}

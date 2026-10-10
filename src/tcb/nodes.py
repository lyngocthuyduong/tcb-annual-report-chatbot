"""Typed nodes with provenance. One node never spans two printed pages."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field

from . import conf, parse
from .pdf import Half


@dataclass
class Node:
    chunk_id: str
    text: str
    document_id: str
    pdf_page_index: int
    side: str
    printed_page_number: int | None
    page_number_source: str
    section: str
    content_type: str                  # narrative | kpi | table | glossary
    bbox: list[float]
    table_id: str | None = None
    row_range: list[int] | None = None
    form_code: str | None = None
    unit_header: str | None = None
    continuation: bool = False
    quality: str | None = None         # tables: high | low (low = layout not cleanly resolved; narrative view kept too)
    rows: list[dict] | None = None     # structured FS rows (tables only)

    def meta(self) -> dict:
        d = asdict(self); d.pop("text"); d.pop("rows"); return d


def _cid(*parts) -> str:
    return hashlib.sha1("|".join(map(str, parts)).encode("utf-8")).hexdigest()[:16]


def _words(t: str) -> int:
    return len(t.split())


def _split_sentences(t: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?;:])\s+(?=[A-ZÀ-Ỵ0-9“\"•])", t) if s.strip()]


def _hard_split(t: str, budget: int) -> list[str]:
    w = t.split()
    return [t] if len(w) <= budget else [" ".join(w[i:i + budget]) for i in range(0, len(w), budget)]


def _pack(texts: list[str], budget: int) -> list[str]:
    """Pack consecutive texts into chunks <= budget words; split oversize texts on sentences."""
    chunks, buf, n = [], [], 0
    def flush():
        nonlocal buf, n
        if buf: chunks.append("\n".join(buf)); buf, n = [], 0
    for t in texts:
        parts = [t] if _words(t) <= budget else _split_sentences(t) or [t]
        parts = [q for p in parts for q in _hard_split(p, budget)]
        for p in parts:
            w = _words(p)
            if n + w > budget and buf: flush()
            buf.append(p); n += w
    flush()
    return chunks


def half_nodes(h: Half, section: str, chunk_words: int | None = None) -> tuple[list[Node], list[tuple[str, str]]]:
    """-> (nodes, glossary entries) for one half-page."""
    nodes: list[Node] = []
    gloss: list[tuple[str, str]] = []
    if not h.blocks: return nodes, gloss
    base = dict(document_id=conf.DOC_ID, pdf_page_index=h.pdf_page_index, side=h.side,
                printed_page_number=h.printed_page, page_number_source=h.page_source, section=section)
    crumb = f"[{section}] " if section else ""
    ymask: list[tuple[float, float]] = []

    # glossary page(s)
    if any(b.text.startswith("Thuật ngữ") or "Thuật ngữ" in b.text[:20] for b in h.blocks):
        gl = parse.glossary(h)
        if gl:
            gloss += gl
            for i, (t, d) in enumerate(gl):
                nodes.append(Node(_cid(h.pdf_page_index, h.side, "g", t), f"{t}: {d}", content_type="glossary",
                                  bbox=[0, 0, 0, 0], row_range=[i, i], **base))
            return nodes, gloss

    # FS tables
    if section == parse.SECTION_FS:
        tbs, ymask = parse.tables(h)
        for tb in tbs:
            unit = next((u for u in tb.units if u), None)
            head = f"{tb.title} ({tb.form_code or 'bảng'})" + (f" — đơn vị: {unit}" if unit else "")
            for k in range(0, len(tb.rows), conf.TABLE_ROWS_PER_CHUNK):
                part = tb.rows[k:k + conf.TABLE_ROWS_PER_CHUNK]
                lines = []
                for r in part:
                    cells = "; ".join(f"{tb.headers[i]}: {c}" for i, c in enumerate(r.cells) if c is not None)
                    nt = f" [TM {r.note}]" if r.note else ""
                    lines.append(f"{r.label}{nt} | {cells}")
                text = f"{crumb}{head}\n" + "\n".join(lines)
                nodes.append(Node(_cid(h.pdf_page_index, h.side, tb.table_id, k), text, content_type="table",
                                  bbox=[0, part[0].y, 0, part[-1].y], table_id=tb.table_id,
                                  row_range=[k, k + len(part) - 1], form_code=tb.form_code, unit_header=unit,
                                  continuation=tb.continuation, quality="high" if tb.reliable else "low",
                                  rows=[{"label": r.label, "note": r.note, "indent": round(r.indent),
                                         "cells": dict(zip(tb.headers, r.cells))} for r in part], **base))

    # narrative + KPI from the rest
    blocks = [b for b in h.blocks if not any(a <= (b.bbox[1] + b.bbox[3]) / 2 <= z for a, z in ymask)]
    buf: list[str] = []
    order = 0

    def flush_buf():
        nonlocal order
        for t in _pack(buf, chunk_words or conf.CHUNK_MAX_WORDS):
            if _words(t) >= 3:
                nodes.append(Node(_cid(h.pdf_page_index, h.side, "n", order), crumb + t, content_type="narrative",
                                  bbox=[0, 0, 0, 0], **base))
                order += 1
        buf.clear()

    for g in parse.groups(blocks):
        txt = re.sub(r"\s*\n\s*", " ", g.text).strip()
        if g.kind == "kpi":
            flush_buf()
            x0, y0, x1, y1 = g.bbox
            nodes.append(Node(_cid(h.pdf_page_index, h.side, "k", round(x0), round(y0)), crumb + " | ".join(
                s.strip() for s in g.text.split("\n") if s.strip()), content_type="kpi",
                bbox=[round(x0 - h.x_off, 1), round(y0, 1), round(x1 - h.x_off, 1), round(y1, 1)], **base))
        else:
            buf.append(txt)
    flush_buf()
    return nodes, gloss


def build_nodes(halves: list[Half], chunk_words: int | None = None) -> tuple[list[Node], dict[str, dict]]:
    by_page = {(h.pdf_page_index, h.side): h for h in halves}
    nodes: list[Node] = []
    gloss: dict[str, dict] = {}
    for h in halves:
        sec = parse.section_of(by_page, h.pdf_page_index)
        ns, gl = half_nodes(h, sec, chunk_words)
        nodes += ns
        for t, d in gl: gloss[t] = {"definition": d, "printed_page": h.printed_page}
    return nodes, gloss


def dump_jsonl(nodes: list[Node], path) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for n in nodes:
            f.write(json.dumps(asdict(n), ensure_ascii=False) + "\n")

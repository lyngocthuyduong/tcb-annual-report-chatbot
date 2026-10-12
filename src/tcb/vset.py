"""Dev/val dataset check: every answerable item's gold facts (or, for calc items, its inputs) must appear in the parsed
text of its gold/alt printed pages. The sets themselves are written by hand in data/dev.json and data/val.json."""
from __future__ import annotations

from . import cite, ev


def _holds(text: str, fact: str) -> bool:
    for alt in fact.split(" | "):
        fn = {t for t in cite.num_tokens(alt) if not cite._trivial(t)}
        if fn and fn <= cite.num_tokens(text): return True
        if not fn and ev._fold(alt) in ev._fold(text): return True
    return False


def check(set_path: str, nodes) -> list[str]:
    bad = []
    for it in ev.load_items(set_path):
        if not it.answerable:
            continue
        pages = set(it.gold_printed_pages) | set(it.alt_pages)
        text = " ".join(n.text for n in nodes if n.printed_page_number in pages)
        miss = [f for f in it.gold_facts if not _holds(text, f)]
        if it.type == "calc":                                # the result is computed; its INPUTS must be on the page
            src = {t for t in cite.num_tokens(it.gold_answer) if not cite._trivial(t)}
            for f in it.gold_facts: src -= cite.num_tokens(f)
            miss = [f"input {t}" for t in sorted(src) if t not in cite.num_tokens(text)]
        if not pages: bad.append(f"{it.id}: answerable but no gold page")
        elif miss: bad.append(f"{it.id}: {miss} not on printed pages {sorted(pages)}")
    return bad

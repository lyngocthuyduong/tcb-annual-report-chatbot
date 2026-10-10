"""Dev/val candidate sets. Everything here is a DRAFT (verified=false) until a human checks it against the PDF.
gen    : table items (auto, from high-quality statement tables) + data/hand.json -> data/dev.json, data/val.json
export : set JSON -> CSV (utf-8-sig, opens in Excel) for review
apply  : reviewed CSV -> set JSON  (action Y = verified, D = drop, edits to the editable columns are kept)
check  : every answerable item's gold facts must appear in the text of the nodes on its gold pages"""
from __future__ import annotations

import csv
import json
import random
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

from . import cite, ev, num

PAGES = range(297, 303)             # B01-B04 statements (high-quality tables only)
SEP = " ;; "                        # separator for list columns in the CSV
COLS = ["id", "action", "type", "category", "question", "gold_answer", "gold_printed_pages", "alt_pages", "gold_facts", "verified", "notes"]


def _hdr_phrase(h: str) -> tuple[str, str]:
    """('tại ngày 31/12/2025', 'ngày 31/12/2024') or ('năm 2025', 'năm 2024')."""
    p = num.parse_period(h)
    return (f"tại ngày {p.label}", f"ngày {p.label}") if p.kind == "date" else (f"năm {p.label}", f"năm {p.label}")


def good_label(lab: str) -> bool:
    """A label that makes a self-contained question: no leaked header text, no unit in the label, not a merged line."""
    bad = ("Triệu đồng", "Thuyết minh", "(đồng", "(*)", "như sau")
    return (8 <= len(lab) <= 90 and lab[0].isupper() and not any(b in lab for b in bad)
            and not lab.endswith(" năm") and not lab.startswith(("Tăng ", "Giảm ", "(Tăng)", "(Giảm)")) and not num.parse_period(lab).kind != "none")


def table_rows(nodes) -> list[dict]:
    """Eligible rows: unique label across the statements, two positive numeric cells (current, previous)."""
    rows = []
    for n in nodes:
        if n.content_type != "table" or n.quality != "high" or n.printed_page_number not in PAGES:
            continue
        for r in n.rows:
            hs = list(r["cells"])
            if len(hs) != 2 or any(r["cells"][h] is None for h in hs):
                continue
            try:
                cur, prev = num.parse_vn(r["cells"][hs[0]]), num.parse_vn(r["cells"][hs[1]])
            except num.NumError:
                continue
            lab = r["label"].strip()
            if cur is None or prev is None or cur <= 0 or prev <= 0 or not good_label(lab):
                continue
            if {r["cells"][hs[0]], r["cells"][hs[1]]} == {"2025", "2024"}:
                continue
            rows.append({"label": lab, "page": n.printed_page_number, "h_cur": hs[0], "h_prev": hs[1],
                         "cur": r["cells"][hs[0]], "prev": r["cells"][hs[1]], "cur_v": cur, "prev_v": prev,
                         "unit": "triệu đồng"})
    cnt: dict[str, int] = {}
    for r in rows: cnt[r["label"].lower()] = cnt.get(r["label"].lower(), 0) + 1
    return [r for r in rows if cnt[r["label"].lower()] == 1]


def growth(cur: Decimal, prev: Decimal) -> str:
    """Independent of calc.py on purpose: plain Decimal, 1 dp."""
    return num.fmt(((cur - prev) / prev * 100).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP), 1)


def table_items(nodes, seed: int = 7, n_single=24, n_calc=12, n_follow=5) -> list[dict]:
    rng = random.Random(seed)
    pool = sorted(table_rows(nodes), key=lambda r: (r["page"], r["label"]))
    rng.shuffle(pool)
    out = []
    for i, r in enumerate(pool[:n_single]):
        w, _ = _hdr_phrase(r["h_cur"])
        out.append({"id": f"t-{i+1:02d}", "category": "financial_statements", "type": "single",
                    "question": f"{r['label']} của Techcombank {w} là bao nhiêu triệu đồng?",
                    "gold_answer": f"{r['cur']} triệu đồng.", "gold_printed_pages": [r["page"]], "gold_facts": [r["cur"]]})
    rest = pool[n_single:]
    calc_pool = [r for r in rest if Decimal("0.5") <= abs((r["cur_v"] - r["prev_v"]) / r["prev_v"] * 100) <= 300]
    for i, r in enumerate(calc_pool[:n_calc]):
        w, ref = _hdr_phrase(r["h_cur"])
        _, ref_prev = _hdr_phrase(r["h_prev"])
        out.append({"id": f"c-{i+1:02d}", "category": "financial_statements", "type": "calc",
                    "question": f"{r['label']} {w} tăng hay giảm bao nhiêu phần trăm so với {ref_prev}?",
                    "gold_answer": f"{growth(r['cur_v'], r['prev_v'])}% ({r['cur']} so với {r['prev']} triệu đồng).",
                    "gold_printed_pages": [r["page"]], "gold_facts": [growth(r["cur_v"], r["prev_v"])]})
    used = {id(r) for r in calc_pool[:n_calc]}
    for i, r in enumerate([r for r in rest if id(r) not in used][:n_follow]):
        w, _ = _hdr_phrase(r["h_cur"])
        _, ref_prev = _hdr_phrase(r["h_prev"])
        q0 = f"{r['label']} của Techcombank {w} là bao nhiêu triệu đồng?"
        out.append({"id": f"f-{i+1:02d}", "category": "conversation", "type": "followup",
                    "history": [{"question": q0, "answer": f"{r['label']} {w} là {r['cur']} triệu đồng. [tr. {r['page']}]"}],
                    "question": f"Còn {ref_prev} thì sao?", "gold_answer": f"{r['prev']} triệu đồng.",
                    "gold_printed_pages": [r["page"]], "gold_facts": [r["prev"]]})
    for it in out:
        it["verified"] = False
        it["notes"] = "AUTO from parsed statement table; verify the figure against the PDF page"
    return out


def split(items: list[dict], seed: int = 7) -> tuple[list[dict], list[dict]]:
    """Deterministic ~2:1 dev/val, stratified by (type, answerable)."""
    rng = random.Random(seed)
    groups: dict[tuple, list[dict]] = {}
    for it in items:
        groups.setdefault((it.get("type"), it.get("answerable", True), it.get("category") == "financial_statements"), []).append(it)
    dev, val = [], []
    for k in sorted(groups, key=str):
        g = sorted(groups[k], key=lambda x: x["id"]); rng.shuffle(g)
        for j, it in enumerate(g):
            (val if j % 3 == 2 else dev).append(it)
    key = lambda x: x["id"]
    return sorted(dev, key=key), sorted(val, key=key)


def gen(nodes, hand_path="data/hand.json", out_dir="data", seed: int = 7) -> dict:
    hand = json.loads(Path(hand_path).read_text(encoding="utf-8"))
    for h in hand:
        h.setdefault("verified", False)
        h.setdefault("notes", "DRAFT by assistant; verify against the PDF" if h.get("answerable", True)
                     else "DRAFT unanswerable; absence checked by text search of the parsed report only")
    items = table_items(nodes, seed) + hand
    dev, val = split(items, seed)
    for name, xs in (("dev", dev), ("val", val)):
        Path(out_dir, f"{name}.json").write_text(json.dumps(xs, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        ev.load_items(Path(out_dir, f"{name}.json"))        # schema check
    return {"dev": len(dev), "val": len(val), "total": len(items)}


# ------------------------------------------------------------------ review round trip
def export_csv(set_path: str, csv_path: str) -> int:
    items = json.loads(Path(set_path).read_text(encoding="utf-8"))
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, COLS); w.writeheader()
        for it in items:
            w.writerow({"id": it["id"], "action": "", "type": it.get("type", ""), "category": it.get("category", ""),
                        "question": it["question"], "gold_answer": it.get("gold_answer", ""),
                        "gold_printed_pages": ",".join(map(str, it.get("gold_printed_pages", []))),
                        "alt_pages": ",".join(map(str, it.get("alt_pages", []))),
                        "gold_facts": SEP.join(it.get("gold_facts", [])), "verified": str(bool(it.get("verified"))).lower(),
                        "notes": it.get("notes", "")})
    return len(items)


def _ints(s: str) -> list[int]:
    return [int(x) for x in s.replace(";", ",").split(",") if x.strip()]


def apply_csv(set_path: str, csv_path: str) -> dict:
    items = {it["id"]: it for it in json.loads(Path(set_path).read_text(encoding="utf-8"))}
    order = list(items)
    stat = {"verified": 0, "dropped": 0, "edited": 0, "untouched": 0}
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            it = items.get(r["id"])
            act = (r.get("action") or "").strip().upper()
            if it is None or not act:
                stat["untouched"] += 1; continue
            if act == "D":
                del items[r["id"]]; stat["dropped"] += 1; continue
            if act != "Y":
                raise ValueError(f"{r['id']}: action must be Y, D or empty, got {act!r}")
            before = json.dumps(it, sort_keys=True, ensure_ascii=False)
            it["question"] = r["question"].strip(); it["gold_answer"] = r["gold_answer"].strip()
            it["gold_printed_pages"] = _ints(r["gold_printed_pages"]); it["alt_pages"] = _ints(r["alt_pages"])
            it["gold_facts"] = [x.strip() for x in r["gold_facts"].split(";;") if x.strip()]
            it["verified"] = True
            if r.get("notes"): it["notes"] = r["notes"].strip()
            if not it.get("alt_pages"): it.pop("alt_pages", None)
            stat["verified"] += 1
            stat["edited"] += json.dumps(it, sort_keys=True, ensure_ascii=False) != before.replace('"verified": false', '"verified": true')
    out = [items[i] for i in order if i in items]
    Path(set_path).write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    ev.load_items(set_path)                                  # re-validate (e.g. unanswerable must have no pages)
    return stat


# ------------------------------------------------------------------ gold-on-page check
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

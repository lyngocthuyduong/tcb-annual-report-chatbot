"""REGRESSION on the REAL parsed PDF (integration; no models): every published q10 gold fact must be present, as text,
in a node that carries the gold printed page. Guards page mapping + extraction, independent of retrieval/LLM."""
import pytest

from tcb import cite, ev, vi

pytestmark = pytest.mark.integ
ITEMS = [i for i in ev.load_items("data/q10.json") if i.answerable]


def holds(node_text: str, fact: str) -> bool:
    for alt in fact.split(" | "):
        fn = {t for t in cite.num_tokens(alt) if not cite._trivial(t)}
        if fn and fn <= cite.num_tokens(node_text): return True
        if not fn and ev._fold(alt) in ev._fold(node_text): return True
    return False


@pytest.mark.parametrize("it", ITEMS, ids=lambda i: i.id)
def test_gold_facts_are_on_gold_pages(doc, it):
    pages = set(it.gold_printed_pages) | set(it.alt_pages)
    text = " ".join(n.text for n in doc["nodes"] if n.printed_page_number in pages)
    missing = [f for f in it.gold_facts if not holds(text, f)]
    assert not missing, f"{it.id}: {missing} not found on printed pages {sorted(pages)}"


def test_printed_to_pdf_page_formula(doc):
    for n in doc["nodes"]:
        p = n.printed_page_number
        if p is not None:
            assert n.pdf_page_index == (p + 2) // 2 if p % 2 else n.pdf_page_index == p // 2 + 1, (p, n.pdf_page_index)
            assert n.side == ("right" if p % 2 else "left")


def test_every_node_has_citable_page_except_cover(doc):
    none = [n for n in doc["nodes"] if n.printed_page_number is None]
    assert all(n.pdf_page_index == 1 for n in none), "only the cover may lack a printed page"


def test_financial_statement_figures_golden(doc):
    b02 = [n for n in doc["nodes"] if n.printed_page_number == 297 and n.content_type == "table"]
    t = " ".join(n.text for n in b02)
    for f in ("1.192.344.137", "978.798.549", "4.360.821", "3.384.770"):
        assert f in t
    row = next(r for n in b02 for r in n.rows if r["label"].startswith("Tiền mặt"))
    assert row["note"] == "5" and list(row["cells"].values()) == ["4.360.821", "3.384.770"]
    assert "(181.150)" in " ".join(n.text for n in doc["nodes"] if n.content_type == "table")


def test_quantities_extracted_from_real_table(doc):
    from tcb import calc, num
    n = next(n for n in doc["nodes"] if n.printed_page_number == 297 and "TỔNG TÀI SẢN" in n.text and n.content_type == "table")
    from dataclasses import asdict
    k = [0]
    def nid(): k[0] += 1; return f"Q{k[0]}"
    qs = [q for q in num.from_node(asdict(n), nid) if q.label.startswith("TỔNG TÀI SẢN")]
    cur = next(q for q in qs if q.period.year == 2025); prev = next(q for q in qs if q.period.year == 2024)
    assert (cur.dim, cur.scale, cur.page) == ("VND", 10**6, 297) and str(cur.value) == "1192344137"
    assert calc.growth_pct(cur, prev).text == "21,8%"            # equals the report's own KPI "Tăng 21,8%" (p.5)


def test_low_quality_tables_yield_no_quantities(doc):
    from dataclasses import asdict
    from tcb import num
    low = [n for n in doc["nodes"] if n.content_type == "table" and n.quality == "low"]
    assert low and all(num.from_node(asdict(n), lambda: "Q") == [] for n in low)

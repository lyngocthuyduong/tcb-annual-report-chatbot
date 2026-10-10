from conftest import text_of


def test_e1_page_map(doc):
    i = doc["info"]
    assert i["conflicts"] == [] and i["pdf_pages"] == 197 and i["halves"] == 393
    assert i["footer"] == 380 and i["inferred"] == 13        # cover + 6 divider spreads


def test_inferred_pages_are_dividers(doc):
    inferred = sorted(h.printed_page for h in doc["halves"] if h.page_source == "inferred")
    assert inferred == [1, 8, 9, 24, 25, 44, 45, 100, 101, 144, 145, 290, 291]


def test_golden_halves(doc):
    H = doc["H"]
    assert H[(3, "left")].printed_page == 4 and H[(3, "right")].printed_page == 5
    assert "302 chi nhánh" in text_of(H[(3, "left")]) and "302 chi nhánh" not in text_of(H[(3, "right")])
    assert "1.192" in text_of(H[(3, "right")]) and "1.192" not in text_of(H[(3, "left")])
    assert H[(30, "right")].printed_page == 59 and "328,1" in text_of(H[(30, "right")])
    assert H[(149, "right")].printed_page == 297 and "TỔNG TÀI SẢN" in text_of(H[(149, "right")])
    assert H[(194, "left")].printed_page == 386 and "CASA" in text_of(H[(194, "left")])
    assert H[(194, "right")].printed_page == 387 and "RBG" in text_of(H[(194, "right")])


def test_linear_map_all_halves(doc):
    from tcb import pdf
    for h in doc["halves"]:
        if h.side == "single": continue
        assert h.printed_page == pdf.linear_map(h.pdf_page_index)[0 if h.side == "left" else 1]


def test_straddle_rare(doc):
    assert doc["info"]["straddle_words"] <= 10

from tcb import vi


def test_numbers_kept_whole():
    t = vi.tokens("Tỷ lệ an toàn vốn 14,6%; Tổng tài sản 1.192 nghìn tỷ đồng; CASA 40,4%; 1.192.344.137")
    for x in ["an", "14,6%", "14.6%", "1.192", "1192", "40,4%", "casa", "1192344137", "nghìn", "tỷ"]:
        assert x in t, x


def test_forms_dates_and_abbrev():
    t = vi.tokens("B05/TCTD-HN 31/12/2025 tăng 26,9% N/N")
    assert "31/12/2025" in t and "b05/tctd-hn" in t and "n/n" in t


def test_diacritic_fold_extra_token():
    t = vi.tokens("tiền gửi không kỳ hạn")
    assert "tiền" in t and "tien" in t and "gui" in t


def test_negative_paren_number_tokenised():
    assert "181.150" in vi.tokens("(181.150)") and "181150" in vi.tokens("(181.150)")


def test_trailing_decimal_zero_is_the_same_number():
    from tcb import ev
    assert vi.canon_num("16,0%") == vi.canon_num("16%") == "16%" and vi.canon_num("14,60") == "14.6"
    it = ev.Item(id="x", question="q", gold_facts=["16"], gold_printed_pages=[5])
    assert ev.facts_ok(it, "Thị phần đạt 16,0%.")[0] is True

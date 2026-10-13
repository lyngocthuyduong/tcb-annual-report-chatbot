import re


def test_every_node_has_printed_page(doc):
    assert all(n.printed_page_number is not None for n in doc["nodes"])


def test_ids_unique_and_stable(doc):
    from tcb import nodes, pdf
    ids = [n.chunk_id for n in doc["nodes"]]
    assert len(ids) == len(set(ids))
    halves, _ = pdf.load("data/ar.pdf")
    again, _ = nodes.build_nodes(halves)
    assert [n.chunk_id for n in again] == ids and [n.text for n in again] == [n.text for n in doc["nodes"]]


def test_no_furniture_in_nodes(doc):
    assert not any("Báo cáo thường niên 2025" in n.text for n in doc["nodes"] if n.content_type != "glossary")


def test_sample_gold_facts_on_gold_pages(doc):
    ns = doc["nodes"]
    def has(page, pat): return any(n.printed_page_number == page and re.search(pat, n.text) for n in ns)
    assert has(4, r"302 chi nhánh") and has(4, r"BB") and has(4, r"BB-")
    assert has(5, r"1\.192") and has(5, r"40,4%") and has(5, r"1,13%") and has(5, r"53,4")
    assert has(59, r"328,1")
    assert has(386, r"CASA: Tiền gửi không kỳ hạn") and has(387, r"RBG: Khối Ngân hàng Bán lẻ")


def test_nodes_never_span_pages(doc):
    assert all(isinstance(n.printed_page_number, int) for n in doc["nodes"])


def test_content_types_present(doc):
    types = {n.content_type for n in doc["nodes"]}
    assert types == {"narrative", "kpi", "table", "glossary"}


def test_page_headings_give_each_node_its_context(doc):
    """A KPI tile alone ('Dư nợ vay | 328,1') does not say whose figure it is; the page title does (it sits on the facing page)."""
    ns = doc["nodes"]
    def on(page, pat): return [n for n in ns if n.printed_page_number == page and re.search(pat, n.text)]
    assert (x := on(59, r"328,1|475,8")) and all("Khối Ngân hàng Bán lẻ (RBG)" in n.text for n in x)
    assert (x := on(78, r"7\.109")) and all("TCBS" in n.text for n in x)
    assert on(84, r".") and not any("TCBS" in n.text for n in on(84, r".")), "the next subsidiary replaces the carried title"
    assert (x := on(148, r"9 thành viên")) and all("Hội đồng quản trị" in n.text for n in x)
    assert (x := on(302, r"Mua sắm tài sản cố định")) and all("Báo cáo lưu chuyển tiền tệ hợp nhất" in n.text for n in x)
    assert on(387, r"Danh mục thuật ngữ viết tắt\] IRRBB: Rủi ro lãi suất trên sổ ngân hàng")


def test_kpi_tiles_are_not_glued_by_a_paragraph(doc):
    kpi = [n.text for n in doc["nodes"] if n.content_type == "kpi" and n.printed_page_number == 59]
    assert any("Số dư tiền gửi | 475,8" in t for t in kpi) and any("Tổng thu nhập hoạt động | 28,3" in t for t in kpi)


def test_two_column_prose_is_read_column_by_column(doc):
    """A heading spanning both columns used to merge them; sorting by y then interleaved paragraphs of the two columns."""
    t = " ".join(n.text for n in doc["nodes"] if n.printed_page_number == 53 and n.content_type == "narrative")
    assert "thêm 23,0% N/N, đạt 776,5 nghìn tỷ đồng" in t and "N/N, với cùng kỳ" not in t
    assert "phiên bản nâng cấp được thiết kế nhằm mang lại nhiều giá trị hơn" in t


def test_number_only_tiles_are_not_indexed(doc):
    """Chart ticks and contents-page numbers carry no meaning alone and would match numeric questions by accident."""
    kpi = [n.text.split("] ", 1)[-1] for n in doc["nodes"] if n.content_type == "kpi"]
    assert not any(re.fullmatch(r"[\d\s|%,.QqA-Z]+", t) for t in kpi)
    assert all(len(re.findall(r"[^\W\d_]{2,}", t)) >= 2 for t in kpi)


def test_statement_rows_and_headers_are_clean(doc):
    rows = [(n, r) for n in doc["nodes"] if n.content_type == "table" and n.quality == "high" for r in n.rows]
    labels = {r["label"] for _, r in rows}
    # a sub-heading under a wrapped subtotal stays its own row (B04 p.301)
    assert "Lưu chuyển tiền thuần từ hoạt động kinh doanh trước những thay đổi về tài sản và vốn lưu động" in labels
    assert "Những thay đổi về tài sản hoạt động" in labels
    # every column of a high-quality table is a clean period (+ unit); no text leaked from above the table
    hdrs = {h for _, r in rows for h in r["cells"]}
    assert all(re.fullmatch(r"(\d{2}/\d{2}/\d{4}|Năm (19|20)\d{2})( (Triệu đồng|%|Tỷ đồng|Nghìn tỷ đồng|Triệu USD|Đồng))?", h, re.I) for h in hdrs), hdrs
    assert not any(re.search(r"\d{1,3}(\.\d{3}){2,}", lab) for lab in labels)


def test_centred_box_does_not_merge_columns_and_sentences_run_on(doc):
    """p.4: a centred award box over both columns used to bridge them; the sentence '302 chi nhánh và / phòng giao dịch
    tại 29 trên 34 tỉnh thành' wraps from the left column to the right one and must read as one sentence (sq-01)."""
    t = " ".join(n.text for n in doc["nodes"] if n.printed_page_number == 4 and n.content_type == "narrative")
    assert "302 chi nhánh và phòng giao dịch tại 29 trên 34 tỉnh thành" in t
    starts = [n.text.split("] ", 1)[-1][:1] for n in doc["nodes"] if n.content_type == "narrative" and n.section != "Báo cáo tài chính"]
    assert sum(c.islower() for c in starts) <= 10, "continuations are joined to the sentence they finish"


def test_page_form_code_is_never_a_column(doc):
    """The form code printed at the top right of every notes page ('B05/TCTD-HN') sits above the last column; it must not
    become that column's header, and a year row inside the table ('Bên liên quan | 2025 | 2024') is the real header."""
    tables = [n for n in doc["nodes"] if n.content_type == "table"]
    assert not any("TCTD-HN:" in n.text for n in tables)
    p369 = [n for n in tables if n.printed_page_number == 369]
    assert p369 and all(n.quality == "high" for n in p369)
    assert any("Công ty Cổ phần Tập đoàn Masan (i) | Năm 2025 Triệu đồng: 534.314; Năm 2024 Triệu đồng: 710.409" in n.text
               for n in p369)


def test_column_unit_comes_from_the_unit_line_under_the_years(doc):
    """p.351: a tax-rate sentence ('20% (2024: 20%)') above the table must not make the columns '%'."""
    p351 = [n for n in doc["nodes"] if n.content_type == "table" and n.printed_page_number == 351]
    assert p351 and all("Năm 2025 Triệu đồng: 79.856" in n.text or "Số dư đầu năm" not in n.text for n in p351)
    assert all(n.unit_header == "Triệu đồng" for n in p351)


def test_note_tables_are_named_by_their_numbered_heading(doc):
    def heads(p): return {n.text.split("\n")[0] for n in doc["nodes"] if n.content_type == "table" and n.printed_page_number == p}
    assert any("6. TIỀN GỬI TẠI NGÂN HÀNG NHÀ NƯỚC VIỆT NAM" in h for h in heads(328))     # second table on the page
    assert all("41. GIAO DỊCH VỚI CÁC BÊN LIÊN QUAN" in h for h in heads(370))              # carried from an earlier page


def test_statement_page_band_is_not_narrative(doc):
    """'THUYẾT MINH ... (tiếp theo)', the period line and the form code repeat on ~80 pages: furniture, not content."""
    fs = [n for n in doc["nodes"] if n.section == "Báo cáo tài chính" and n.content_type == "narrative"]
    assert not any("HỢP NHẤT (tiếp theo)" in n.text or re.search(r"(^|\n|\] )B05/TCTD-HN($|\n)", n.text) for n in fs)
    assert any(n.text.startswith("[Báo cáo tài chính › Thuyết minh báo cáo tài chính hợp nhất]") for n in fs)

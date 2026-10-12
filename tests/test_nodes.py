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

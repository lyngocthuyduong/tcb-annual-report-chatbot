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

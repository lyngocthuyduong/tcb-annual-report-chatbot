import re

from tcb import parse


def test_b02_rows(doc):
    t, _ = parse.tables(doc["H"][(149, "right")])
    tb = t[0]
    assert tb.form_code == "B02/TCTD-HN" and tb.headers == ["31/12/2025 Triệu đồng", "31/12/2024 Triệu đồng"]
    by = {r.label: r for r in tb.rows}
    assert by["Tiền mặt, vàng"].note == "5" and by["Tiền mặt, vàng"].cells == ["4.360.821", "3.384.770"]
    assert by["TỔNG TÀI SẢN"].cells == ["1.192.344.137", "978.798.549"]
    neg = [r for r in tb.rows if r.label.startswith("Dự phòng rủi ro cấp tín dụng cho các TCTD khác")][0]
    assert neg.cells == ["(181.150)", "(181.150)"] and neg.note == "7.3"


def test_nil_is_dash_not_minus(doc):
    t, _ = parse.tables(doc["H"][(149, "right")])
    r = [x for x in t[0].rows if "giữ đến ngày đáo hạn" in x.label][0]
    assert r.cells == ["-", "6.900.000"] and r.note == "12.2"


def test_glossary_terms(doc):
    g = doc["gloss"]
    assert g["CASA"]["definition"] == "Tiền gửi không kỳ hạn" and g["CASA"]["printed_page"] == 386
    assert g["RBG"]["definition"] == "Khối Ngân hàng Bán lẻ" and g["RBG"]["printed_page"] == 387
    assert g["AGM"]["definition"] == "Đại hội đồng cổ đông thường niên"
    assert len(g) >= 150


def _kpi(doc, page):
    return [n.text for n in doc["nodes"] if n.printed_page_number == page and n.content_type == "kpi"]


def test_kpi_tiles_p5(doc):
    k = " || ".join(_kpi(doc, 5))
    assert re.search(r"Tổng tài sản \| 1\.192 \| Nghìn tỷ đồng \| Tăng 21,8%", k)
    assert "Tỷ lệ CASA | 40,4%" in k and "Tỷ lệ nợ xấu | 1,13%" in k
    assert re.search(r"Tổng thu nhập hoạt động \| 53,4 \| Nghìn tỷ đồng \| Tăng trưởng kép \| 2018–2025: 16,5%", k)


def test_kpi_tile_p59(doc):
    assert any(re.search(r"Dư nợ vay \| 328,1 .*26,9% \(▲\) N/N", t) for t in _kpi(doc, 59))


def test_drawn_direction_triangles_become_markers(doc):
    """'▲ 7% so với năm 2023' is a vector triangle, not text; without it the change has no direction."""
    assert any("Chỉ số Gắn kết Nhân viên | 93% | 7% (▲) so với năm 2023" in t for t in _kpi(doc, 264))
    assert any("40,4% (▲)" in t for t in _kpi(doc, 53))                # a level marked as rising, not '+40,4%'
    assert doc["info"]["direction_arrows"] > 50

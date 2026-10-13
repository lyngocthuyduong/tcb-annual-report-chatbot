"""UNIT: calculator + number parsing (no models, no PDF)."""
from decimal import Decimal as D

import pytest

from tcb import calc, num
from tcb.num import Period, Quantity

pytestmark = pytest.mark.unit


def Q(qid, v, dim="VND", scale=10**6, kind="date", year=2025, label="x"):
    return Quantity(qid, label, D(v), dim, scale, Period(kind, year, str(year)), "c", 1)


@pytest.mark.parametrize("s,v", [("1.192.344.137", 1192344137), ("14,6", D("14.6")), ("(181.150)", -181150),
                                 ("40,4%", D("40.4")), ("328,1", D("328.1")), ("0", 0), ("1.192", 1192)])
def test_parse_vn(s, v):
    assert num.parse_vn(s) == D(v)


def test_parse_nil_and_bad():
    assert num.parse_vn("-") is None
    for bad in ("abc", "1,2,3", "12.34", "1.2.3"):
        with pytest.raises(num.NumError): num.parse_vn(bad)


@pytest.mark.parametrize("s,u", [("Triệu đồng", ("VND", 10**6)), ("Nghìn tỷ đồng", ("VND", 10**12)), ("tỷ đồng", ("VND", 10**9)),
                                 ("31/12/2025 Triệu đồng", ("VND", 10**6)), ("%", ("%", 1)), ("lần", ("x", 1))])
def test_parse_unit(s, u):
    assert num.parse_unit(s) == u


def test_period():
    assert num.parse_period("31/12/2025 Triệu đồng") == Period("date", 2025, "31/12/2025")
    assert num.parse_period("Năm 2024").kind == "year"
    assert num.parse_period("Triệu đồng").kind == "none"


def test_fmt_vn():
    assert num.fmt(D("1192344137")) == "1.192.344.137"
    assert num.fmt(D("21.8181"), 1) == "21,8"
    assert num.fmt(D("-181150")) == "-181.150"
    assert num.fmt(D("2.25"), 1) == "2,3"  # half-up, not banker's


def test_growth_golden():
    # B02 total assets 31/12/2025 vs 31/12/2024 (triệu đồng) -> report's own "Tăng 21,8%"
    g = calc.growth_pct(Q("Q1", "1192344137"), Q("Q2", "978798549", year=2024))
    assert g.text == "21,8%" and g.formula == "(Q1-Q2)/|Q2|*100"


def test_growth_mixed_scales_ok():
    g = calc.growth_pct(Q("Q1", "1192", scale=10**12), Q("Q2", "978798549", year=2024))
    assert round(g.value, 1) == D("21.8")


@pytest.mark.parametrize("a,b,exc", [
    (Q("a", 1, "%", 1), Q("b", 1, year=2024), calc.UnitMismatch),
    (Q("a", 1), Q("b", 0, year=2024), calc.ZeroDenominator),
    (Q("a", 1), Q("b", 1, year=2025), calc.PeriodMismatch),           # not later
    (Q("a", 1), Q("b", 1, kind="year", year=2024), calc.PeriodMismatch),  # kind differs
    (Q("a", 1, kind="none", year=None), Q("b", 1, kind="none", year=None), calc.PeriodMismatch),
])
def test_growth_errors(a, b, exc):
    with pytest.raises(exc): calc.growth_pct(a, b)


def test_diff_pp_and_money():
    d = calc.diff(Q("a", "40.4", "%", 1), Q("b", "38.0", "%", 1, year=2024))
    assert (d.value, d.dim) == (D("2.4"), "pp") and d.text == "2,40 điểm phần trăm"
    m = calc.diff(Q("a", "10"), Q("b", "4", scale=10**9))
    assert m.value == D("-3990") and m.dim == "VND"
    with pytest.raises(calc.UnitMismatch): calc.diff(Q("a", 1), Q("b", 1, "%", 1))


def test_ratio_and_share():
    assert calc.ratio(Q("a", 6), Q("b", 3)).value == D(2)
    with pytest.raises(calc.ZeroDenominator): calc.ratio(Q("a", 6), Q("b", 0))
    with pytest.raises(calc.UnitMismatch): calc.ratio(Q("a", 6), Q("b", 3, "%", 1))
    s = calc.share_pct(Q("a", 25), Q("b", 100))
    assert s.value == D(25)
    with pytest.raises(calc.PeriodMismatch): calc.share_pct(Q("a", 25), Q("b", 100, year=2024))
    with pytest.raises(calc.ZeroDenominator): calc.share_pct(Q("a", 25), Q("b", 0))


def test_cagr():
    c = calc.cagr(Q("a", 100, year=2020), Q("b", 200, year=2025))
    assert round(c.value, 2) == D("14.87")
    with pytest.raises(calc.PeriodMismatch): calc.cagr(Q("a", 100), Q("b", 200))
    with pytest.raises(calc.InvalidInput): calc.cagr(Q("a", 0, year=2020), Q("b", 200, year=2025))


def test_sum_and_convert():
    s = calc.sum_([Q("a", 1), Q("b", 2), Q("c", 3)])
    assert s.value == D(6)
    with pytest.raises(calc.InvalidInput): calc.sum_([])
    with pytest.raises(calc.PeriodMismatch): calc.sum_([Q("a", 1), Q("b", 2, year=2024)])
    assert calc.convert(Q("a", "4360821"), "tỷ đồng").value == D("4360.821")
    with pytest.raises(calc.UnitMismatch): calc.convert(Q("a", 1, "%", 1), "tỷ đồng")
    with pytest.raises(calc.InvalidInput): calc.convert(Q("a", 1), "mét")


def test_run_dispatch():
    assert calc.run("ratio", [Q("a", 6), Q("b", 3)]).value == D(2)
    for op, qs in (("growth_pct", [Q("a", 1)]), ("nope", []), ("convert", [Q("a", 1)])):
        with pytest.raises(calc.InvalidInput): calc.run(op, qs)


def test_literal_periods_compare_by_year_tables_stay_strict():
    a = num.Quantity("L1", "nhân sự", D(12705), "count", 1, num.Period("date", 2025, "31/12/2025"), "c", 52)
    b = num.Quantity("L2", "nhân sự", D(11848), "count", 1, num.Period("year", 2024, "2024"), "c", 52)
    assert calc.diff(a, b).text == "857"
    with pytest.raises(calc.PeriodMismatch):
        calc.diff(Q("a", 1), Q("b", 1, kind="year", year=2024))


def test_growth_and_cagr_follow_the_periods_not_the_argument_order():
    new, old = Q("L2", 824104, scale=10**9, kind="year", year=2025), Q("L1", 683028, scale=10**9, kind="year", year=2024)
    assert calc.run("growth_pct", [old, new]).text == calc.run("growth_pct", [new, old]).text == "20,7%"
    start, end = Q("Q1", 100, kind="year", year=2018), Q("Q2", 200, kind="year", year=2025)
    assert calc.run("cagr", [end, start]).value == calc.run("cagr", [start, end]).value


def test_integer_sum_prints_without_decimal():
    assert calc.run("sum", [Q("Q1", 6343732), Q("Q2", 5779202)]).text == "12.122.934 triệu đồng"


def test_share_needs_the_same_date_not_only_the_same_year():
    a = Quantity("Q1", "x", D(50), "VND", 10**6, Period("date", 2025, "31/03/2025"), "c", 1)
    b = Quantity("Q2", "x", D(100), "VND", 10**6, Period("date", 2025, "31/12/2025"), "c", 1)
    with pytest.raises(calc.PeriodMismatch): calc.run("share_pct", [a, b])
    c = Quantity("Q3", "x", D(100), "VND", 10**6, Period("date", 2025, "31/12/2025"), "c", 1)
    assert calc.run("share_pct", [a.__class__(**{**a.__dict__, "period": c.period}), c]).text == "50,0%"


def test_ratio_does_not_mix_a_balance_with_a_flow():
    with pytest.raises(calc.PeriodMismatch):
        calc.run("ratio", [Q("Q1", 10, kind="date"), Q("Q2", 5, kind="year")])

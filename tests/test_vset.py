"""Dev/val generator. integ: uses the real parsed PDF, no models."""
import json
import shutil

import pytest

from tcb import ev, vset

pytestmark = pytest.mark.integ


def test_gen_is_deterministic_and_valid(doc, tmp_path):
    shutil.copy("data/hand.json", tmp_path / "hand.json")
    a = vset.gen(doc["nodes"], hand_path=tmp_path / "hand.json", out_dir=tmp_path)
    first = (tmp_path / "dev.json").read_bytes(), (tmp_path / "val.json").read_bytes()
    vset.gen(doc["nodes"], hand_path=tmp_path / "hand.json", out_dir=tmp_path)
    assert first == ((tmp_path / "dev.json").read_bytes(), (tmp_path / "val.json").read_bytes())
    dev, val = ev.load_items(tmp_path / "dev.json"), ev.load_items(tmp_path / "val.json")
    assert a["total"] == len(dev) + len(val)
    assert not ({i.id for i in dev} & {i.id for i in val})
    assert all(not i.verified for i in dev + val), "generated items are drafts"
    assert {i.type for i in dev + val} >= {"single", "calc", "followup", "refusal"}
    assert any(i.expect_status == "ambiguous" for i in dev + val)


@pytest.mark.parametrize("name", ["dev", "val"])
def test_committed_sets_gold_is_on_gold_pages(doc, name):
    assert vset.check(f"data/{name}.json", doc["nodes"]) == []


def test_committed_unanswerable_topics_are_absent(doc):
    from tcb import vi
    text = vi.fold(" ".join(n.text for n in doc["nodes"]).lower())
    for k in ("vietcombank", "campuchia", "bitcoin", "momo", "nasdaq", "world cup", "2027", "mb bank"):
        assert k not in text, k


def test_check_catches_a_wrong_fact(doc, tmp_path):
    items = json.loads(open("data/q10.json", encoding="utf-8").read())
    items[3]["gold_facts"] = ["999.999.999"]
    p = tmp_path / "bad.json"; p.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
    bad = vset.check(str(p), doc["nodes"])
    assert len(bad) == 1 and "999.999.999" in bad[0]


def test_independent_growth_matches_calc():
    from decimal import Decimal
    from tcb import calc, num
    mk = lambda i, v, y: num.Quantity(i, "x", Decimal(v), "VND", 10**6, num.Period("year", y, str(y)), "c", 1)
    assert vset.growth(Decimal(53391125), Decimal(46990397)) == calc.growth_pct(mk("a", 53391125, 2025), mk("b", 46990397, 2024)).text.rstrip("%")


def test_csv_roundtrip_verify_drop_edit(tmp_path):
    src = json.loads(open("data/dev.json", encoding="utf-8").read())
    for x in src: x["verified"] = False      # the shipped set is already checked; the test exercises the mechanism from a draft
    p = tmp_path / "s.json"; p.write_text(json.dumps(src, ensure_ascii=False), encoding="utf-8")
    c = tmp_path / "r.csv"
    assert vset.export_csv(str(p), str(c)) == len(src)
    import csv
    rows = list(csv.DictReader(open(c, encoding="utf-8-sig", newline="")))
    keep = next(r for r in rows if r["id"].startswith("t-")); drop = next(r for r in rows if r["id"].startswith("u-"))
    keep["action"], keep["gold_facts"] = "Y", keep["gold_facts"] + vset.SEP + "extra"
    drop["action"] = "D"
    with open(c, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, vset.COLS); w.writeheader(); w.writerows(rows)
    st = vset.apply_csv(str(p), str(c))
    assert st["verified"] == 1 and st["dropped"] == 1
    out = {i.id: i for i in ev.load_items(p)}
    assert out[keep["id"]].verified and out[keep["id"]].gold_facts[-1] == "extra" and drop["id"] not in out
    assert sum(i.verified for i in out.values()) == 1


def test_apply_rejects_bad_action(tmp_path):
    src = json.loads(open("data/dev.json", encoding="utf-8").read())[:2]
    p = tmp_path / "s.json"; p.write_text(json.dumps(src, ensure_ascii=False), encoding="utf-8")
    c = tmp_path / "r.csv"; vset.export_csv(str(p), str(c))
    txt = open(c, encoding="utf-8-sig").read().replace(",,", ",maybe,", 1)
    open(c, "w", encoding="utf-8-sig").write(txt)
    with pytest.raises(ValueError):
        vset.apply_csv(str(p), str(c))


def test_ambiguous_expectation_scoring():
    it = ev.Item(id="a", question="Tỷ lệ đó là bao nhiêu?", answerable=False, expect_status="ambiguous")
    ok = ev.score_item(it, {"status": "ambiguous", "answer": "Bạn hỏi tỷ lệ nào?", "trace": {}}, lambda c: None)
    bad = ev.score_item(it, {"status": "not_in_report", "answer": "Không có.", "trace": {}}, lambda c: None)
    assert ok["passed"] and not bad["passed"] and bad["verdict"] == "refusal:expected_ambiguous"
    s = ev.summarize([ok, bad])
    assert s["refusal"]["n_unanswerable"] == 0, "ambiguous items must not distort refusal metrics"

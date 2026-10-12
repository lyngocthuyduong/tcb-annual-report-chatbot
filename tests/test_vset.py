"""Dev/val sets. integ: uses the real parsed PDF, no models."""
import json

import pytest

from tcb import ev, vset

pytestmark = pytest.mark.integ


@pytest.mark.parametrize("name", ["dev", "val"])
def test_committed_sets_gold_is_on_gold_pages(doc, name):
    assert vset.check(f"data/{name}.json", doc["nodes"]) == []


def test_committed_unanswerable_topics_are_absent(doc):
    from tcb import vi
    text = vi.fold(" ".join(n.text for n in doc["nodes"]).lower())
    for k in ("vietcombank", "campuchia", "bitcoin", "momo", "nasdaq", "world cup", "2027", "mb bank"):
        assert k not in text, k
    assert "moody" not in text                                    # n-u-moody


def test_shipped_sets_are_checked_and_disjoint():
    dev, val = ev.load_items("data/dev.json"), ev.load_items("data/val.json")
    assert all(i.verified for i in dev + val) and not ({i.question for i in dev if not i.history} & {i.question for i in val if not i.history})
    q10 = {x["question"] for x in json.loads(open("data/q10.json", encoding="utf-8").read())}
    assert not (q10 & {i.question for i in dev + val}), "the published 10 never appear in dev/val"


def test_check_catches_a_wrong_fact(doc, tmp_path):
    items = json.loads(open("data/q10.json", encoding="utf-8").read())
    items[3]["gold_facts"] = ["999.999.999"]
    p = tmp_path / "bad.json"; p.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
    bad = vset.check(str(p), doc["nodes"])
    assert len(bad) == 1 and "999.999.999" in bad[0]


def test_ambiguous_expectation_scoring():
    it = ev.Item(id="a", question="Tỷ lệ đó là bao nhiêu?", answerable=False, expect_status="ambiguous")
    ok = ev.score_item(it, {"status": "ambiguous", "answer": "Bạn hỏi tỷ lệ nào?", "trace": {}}, lambda c: None)
    bad = ev.score_item(it, {"status": "not_in_report", "answer": "Không có.", "trace": {}}, lambda c: None)
    assert ok["passed"] and not bad["passed"] and bad["verdict"] == "refusal:expected_ambiguous"
    s = ev.summarize([ok, bad])
    assert s["refusal"]["n_unanswerable"] == 0, "ambiguous items must not distort refusal metrics"

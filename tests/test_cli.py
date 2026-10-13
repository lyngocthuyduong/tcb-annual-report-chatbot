"""UNIT: batch input formats (the graders' question file format is not known in advance)."""
import json

import pytest

from tcb.cli import load_questions

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("name,body", [
    ("a.json", json.dumps([{"id": "sq-01", "question": "A?", "gold_answer": "x"}, {"question": "B?"}])),
    ("b.json", json.dumps({"questions": ["A?", "B?"]})),
    ("c.jsonl", '{"id": "sq-01", "question": "A?"}\n{"question": "B?"}\n'),
    ("d.txt", "﻿A?\n\nB?\n"),
])
def test_batch_accepts_common_question_files(tmp_path, name, body):
    p = tmp_path / name; p.write_text(body, encoding="utf-8")
    qs = load_questions(p)
    assert [q["question"] for q in qs] == ["A?", "B?"] and qs[1]["id"] == "q2" and all(q["id"] for q in qs)


@pytest.mark.parametrize("body", [json.dumps({"items": ["A?"]}), "[]", json.dumps([{"q": "A?"}])])
def test_batch_refuses_a_file_without_questions(tmp_path, body):
    p = tmp_path / "x.json"; p.write_text(body, encoding="utf-8")
    with pytest.raises(SystemExit):
        load_questions(p)


def test_batch_keeps_follow_up_context(tmp_path, monkeypatch):
    """A follow-up in a batch file is asked with its history, or with the earlier turns of its session."""
    from types import SimpleNamespace as NS
    from tcb import cli
    seen = []

    class E:
        def ask(self, q, turns=None):
            seen.append((q, [t.question for t in turns or []]))
            return NS(status="answered", standalone=q, answer="ok", to_dict=lambda trace=False: {"answer": "ok"})
    monkeypatch.setattr(cli, "_engine", lambda: E())
    p = tmp_path / "q.json"
    p.write_text(json.dumps([{"question": "A?", "session": "s"}, {"question": "Còn B?", "session": "s"},
                             {"question": "Còn C?", "history": [{"question": "X?", "answer": "x"}]}]), encoding="utf-8")
    assert cli.cmd_batch(NS(file=str(p), out=str(tmp_path / "o"))) == 0
    assert seen == [("A?", []), ("Còn B?", ["A?"]), ("Còn C?", ["X?"])]


def test_followup_needs_a_word_that_points_back_or_a_fragment():
    from tcb import chat
    t = [chat.Turn("Tổng tài sản 2025?", "Tổng tài sản 2025?", "1.192 nghìn tỷ", "answered")]
    assert chat.is_followup("Còn năm 2024 thì sao?", t) and chat.is_followup("Năm 2024?", t)
    assert not chat.is_followup("Tỷ lệ CASA năm 2025 là bao nhiêu?", t) and not chat.is_followup("Còn năm 2024?", [])

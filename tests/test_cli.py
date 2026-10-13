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

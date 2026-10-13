"""Multi-turn handling: decide if a question is a follow-up, rewrite it into a standalone query (one LLM call,
guarded), and keep a short history. The history is context for understanding the question, never evidence."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from . import cite, conf
from .llm import LLMRequest, LLMResponse, Provider

MAX_TURNS = 4
_MARK = re.compile(r"\b(còn|vậy|thế|nó|đó|này|kia|tương tự|ngoài ra|tiếp|năm trước|năm ngoái)\b", re.I)   # words that point back
SCHEMA = {"type": "object", "properties": {"standalone": {"type": "string"}}, "required": ["standalone"]}
SYS = ("Viết lại CÂU HỎI CUỐI thành một câu hỏi độc lập, đầy đủ chủ thể/chỉ tiêu/kỳ báo cáo dựa trên hội thoại trước. "
       "Giữ nguyên mọi con số và năm có trong câu hỏi cuối. Không trả lời câu hỏi. Chỉ trả JSON {\"standalone\": \"...\"}.")


@dataclass
class Turn:
    question: str
    standalone: str
    answer: str
    status: str


def is_followup(q: str, turns: list[Turn]) -> bool:
    """A word that points back ('Còn năm 2024 thì sao?'), or a fragment of four words or fewer ('Năm 2024?'). A short
    question that names its own subject ('Tỷ lệ CASA năm 2025 là bao nhiêu?') is asked as it is."""
    return bool(turns) and (bool(_MARK.search(q)) or len(q.split()) <= 4)


def history_text(turns: list[Turn], n: int = 2) -> str:
    return "\n".join(f"Người dùng: {t.standalone}\nTrợ lý: {t.answer[:300]}" for t in turns[-n:])


def new_names(standalone: str, q: str, turns: list[Turn]) -> list[str]:
    """Names (words with a capital letter, after the first word) in the rewrite that occur neither in the question nor in
    the conversation: a rewrite that turns 'Còn năm 2024?' into a question about another bank is not faithful."""
    ctx = (conf.DOC_SUBJECT + "\n" + q + "\n" + "\n".join(f"{t.question} {t.standalone} {t.answer}" for t in turns)).lower()
    words = re.findall(r"[^\W\d_][\w&/-]*", standalone)[1:]
    return [w for w in words if any(c.isupper() for c in w) and w.lower() not in ctx]


def rewrite(prov: Provider, q: str, turns: list[Turn], max_tokens: int = 300) -> tuple[str, str, LLMResponse | None]:
    """-> (standalone, mode, response). mode: none | llm | concat.  Provider errors propagate (no silent fallback);
    an unusable/unfaithful rewrite (drops a number of the question, or brings in a name from nowhere) degrades to a
    deterministic concat and says so in the trace. This guards numbers and names only, not the metric."""
    if not is_followup(q, turns):
        return q, "none", None
    r = prov.generate(LLMRequest(SYS, f"Hội thoại trước:\n{history_text(turns)}\n\nCÂU HỎI CUỐI: {q}", SCHEMA, max_tokens))
    try:
        s = str(json.loads(r.text).get("standalone", "")).strip()
    except (ValueError, AttributeError):
        s = ""
    if s and cite.num_tokens(q) <= cite.num_tokens(s) and not new_names(s, q, turns):
        return s, "llm", r
    return f"{turns[-1].standalone} — {q}", "concat", r


@dataclass
class Session:
    """Holds the turns of one conversation (API keeps one per session id)."""
    turns: list[Turn] = field(default_factory=list)

    def add(self, t: Turn) -> None:
        self.turns.append(t)
        del self.turns[:-MAX_TURNS]

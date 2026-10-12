"""Multi-turn handling: decide if a question is a follow-up, rewrite it into a standalone query (one LLM call,
guarded), and keep a short history. The history is context for understanding the question, never evidence."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from . import cite
from .llm import LLMRequest, LLMResponse, Provider

MAX_TURNS = 4
_MARK = re.compile(r"\b(còn|vậy|thế|nó|đó|này|kia|trên|tương tự|so với|ngoài ra|tiếp|năm trước|năm ngoái|cái|những)\b", re.I)
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
    return bool(turns) and (len(q.split()) <= 9 or bool(_MARK.search(q)))


def history_text(turns: list[Turn], n: int = 2) -> str:
    return "\n".join(f"Người dùng: {t.standalone}\nTrợ lý: {t.answer[:300]}" for t in turns[-n:])


def rewrite(prov: Provider, q: str, turns: list[Turn], max_tokens: int = 300) -> tuple[str, str, LLMResponse | None]:
    """-> (standalone, mode, response). mode: none | llm | concat.  Provider errors propagate (no silent fallback);
    an unusable/unfaithful rewrite degrades to a deterministic concat and says so in the trace."""
    if not is_followup(q, turns):
        return q, "none", None
    r = prov.generate(LLMRequest(SYS, f"Hội thoại trước:\n{history_text(turns)}\n\nCÂU HỎI CUỐI: {q}", SCHEMA, max_tokens))
    try:
        s = str(json.loads(r.text).get("standalone", "")).strip()
    except (ValueError, AttributeError):
        s = ""
    if s and cite.num_tokens(q) <= cite.num_tokens(s):
        return s, "llm", r
    return f"{turns[-1].standalone} — {q}", "concat", r


@dataclass
class Session:
    """Holds the turns of one conversation (API keeps one per session id)."""
    turns: list[Turn] = field(default_factory=list)

    def add(self, t: Turn) -> None:
        self.turns.append(t)
        del self.turns[:-MAX_TURNS]

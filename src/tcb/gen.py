"""Plan generation with exactly one repair retry on the SAME provider. No other provider is ever tried."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from . import plan as P
from .llm import GenerationError, LLMRequest, LLMResponse, Provider


@dataclass
class Gen:
    plan: P.Plan
    responses: list[LLMResponse] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)      # semantic problems left after the retry
    repaired: bool = False


def _repair(user: str, prev: str, errs: list[str]) -> str:
    return (f"{user}\n\n[Phản hồi trước của bạn]\n{prev[:2000]}\n\n[Lỗi cần sửa]\n- " + "\n- ".join(errs[:8]) +
            "\n\nTrả lại MỘT đối tượng JSON đúng schema, đã sửa các lỗi trên.")


def generate_plan(prov: Provider, system: str, user: str, max_tokens: int = 1000,
                  check: Callable[[P.Plan], list[str]] | None = None) -> Gen:
    """1 call if the first reply parses and passes `check`; otherwise 1 repair call; never more.
    GenerationError only if no parseable plan exists after the retry. Provider errors propagate untouched."""
    req = LLMRequest(system, user, P.SCHEMA, max_tokens)
    r1 = prov.generate(req)
    plan1, errs = None, []
    try:
        plan1 = P.parse(r1.text)
        errs = check(plan1) if check else []
        if not errs:
            return Gen(plan1, [r1])
    except ValueError as e:
        errs = [f"JSON/schema: {e}"]
    r2 = prov.generate(LLMRequest(system, _repair(user, r1.text, errs), P.SCHEMA, max_tokens))
    try:
        plan2 = P.parse(r2.text)
    except ValueError as e:
        if plan1 is not None:                              # first reply was parseable; keep it, caller drops bad claims
            return Gen(plan1, [r1, r2], errs, True)
        raise GenerationError(prov.name, prov.model, f"invalid JSON after 1 repair retry: {e}")
    return Gen(plan2, [r1, r2], check(plan2) if check else [], True)

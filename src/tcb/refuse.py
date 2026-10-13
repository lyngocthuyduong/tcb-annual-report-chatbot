"""Refusal / evidence-sufficiency policy. Fail closed: an answer survives only through claims that passed cite.check.
Refusal is decided from the plan and the validators only; there is no retrieval-score threshold (DESIGN 17.1)."""
from __future__ import annotations

from dataclasses import dataclass, field

MSG_NOT_IN_REPORT = "Báo cáo thường niên 2025 của Techcombank không có thông tin để trả lời câu hỏi này."
MSG_NOT_VERIFIED = ("Không đối chiếu được câu trả lời với các trang tìm thấy trong báo cáo, nên hệ thống không đưa ra số liệu. "
                    "Bạn có thể hỏi lại cụ thể hơn (chỉ tiêu, kỳ báo cáo).")
MSG_UNVERIFIED = "Lưu ý: một số nội dung bị loại vì không thể đối chiếu với đoạn trích dẫn trong báo cáo."
MSG_AMBIGUOUS = "Câu hỏi chưa đủ rõ. Bạn có thể nói rõ hơn (chỉ tiêu, kỳ báo cáo, đối tượng) không?"


@dataclass
class Decision:
    status: str
    reason: str                                   # machine-readable, goes to the trace
    notes: list[str] = field(default_factory=list)  # Vietnamese lines appended to the answer


def decide(plan_status: str, n_claims: int, n_kept: int, calc_errors: list[str], clarification: str | None,
           problems_left: bool) -> Decision:
    """n_claims = claims in the plan; n_kept = claims that survived validation and calculation."""
    if plan_status == "ambiguous":
        return Decision("ambiguous", "llm_ambiguous", [clarification.strip() if clarification and clarification.strip() else MSG_AMBIGUOUS])
    if plan_status == "not_in_report":
        return Decision("not_in_report", "llm_declined", [MSG_NOT_IN_REPORT])
    if n_kept == 0:
        why = "all_claims_invalid" if n_claims else "no_claims"
        # every sentence failed the checks: the report may well hold the answer, so do not say it does not
        notes = [MSG_NOT_VERIFIED if n_claims else MSG_NOT_IN_REPORT] + ([f"Không thể tính: {e}" for e in calc_errors] if calc_errors else [])
        return Decision("not_in_report", why, notes)
    notes = [f"Không thể tính: {e}" for e in calc_errors]
    if n_kept < n_claims or problems_left:
        notes.append(MSG_UNVERIFIED)
    if plan_status == "partial" or n_kept < n_claims or calc_errors:
        return Decision("partial", "partial" if plan_status == "partial" else "claims_dropped", notes)
    return Decision("answered", "grounded", notes)

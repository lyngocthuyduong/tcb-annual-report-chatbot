"""Answer plan: the only thing the LLM produces. Facts come from evidence ids; arithmetic from calculator ids."""
from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field, ValidationError, field_validator

LLM_STATUSES = ("answered", "partial", "not_in_report", "ambiguous")


class Claim(BaseModel):
    text_vi: str = Field(min_length=1)
    evidence: list[str] = Field(default_factory=list)          # ["E1", ...]
    calc_refs: list[str] = Field(default_factory=list)         # ["C1", ...]


class Calc(BaseModel):
    cid: str
    op: Literal["growth_pct", "diff", "ratio", "share_pct", "cagr", "sum", "convert"]
    inputs: list[str]                                          # ["Q1", "Q2"]
    to: str | None = None                                      # convert only

    @field_validator("cid")
    @classmethod
    def _cid(cls, v):
        if not re.fullmatch(r"C\d+", v): raise ValueError("cid must look like C1")
        return v


class Lit(BaseModel):
    """A number the LLM copies from an evidence block that has no table quantity (narrative, KPI tile). The code checks
    that `number` occurs verbatim in that block before the calculator may use it."""
    lid: str
    evidence: str                                              # "E3"
    number: str                                                # copied verbatim, e.g. "12.705"
    unit: str = ""                                             # "người", "tỷ đồng", "%"
    period: str = ""                                           # "2025" or "31/12/2025"
    label: str = ""

    @field_validator("lid")
    @classmethod
    def _lid(cls, v):
        if not re.fullmatch(r"L\d+", v): raise ValueError("lid must look like L1")
        return v


class Plan(BaseModel):
    status: Literal["answered", "partial", "not_in_report", "ambiguous"]
    sub_questions: list[str] = Field(default_factory=list)
    claims: list[Claim] = Field(default_factory=list)
    literals: list[Lit] = Field(default_factory=list)
    calculations: list[Calc] = Field(default_factory=list)
    clarification_vi: str | None = None


SCHEMA = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": list(LLM_STATUSES)},
        "sub_questions": {"type": "array", "items": {"type": "string"}},
        "claims": {"type": "array", "items": {"type": "object", "properties": {
            "text_vi": {"type": "string"},
            "evidence": {"type": "array", "items": {"type": "string"}},
            "calc_refs": {"type": "array", "items": {"type": "string"}}},
            "required": ["text_vi", "evidence"]}},
        "literals": {"type": "array", "items": {"type": "object", "properties": {
            "lid": {"type": "string"}, "evidence": {"type": "string"}, "number": {"type": "string"},
            "unit": {"type": "string"}, "period": {"type": "string"}, "label": {"type": "string"}},
            "required": ["lid", "evidence", "number", "unit", "period"]}},
        "calculations": {"type": "array", "items": {"type": "object", "properties": {
            "cid": {"type": "string"},
            "op": {"type": "string", "enum": ["growth_pct", "diff", "ratio", "share_pct", "cagr", "sum", "convert"]},
            "inputs": {"type": "array", "items": {"type": "string"}},
            "to": {"type": "string"}},
            "required": ["cid", "op", "inputs"]}},
        "clarification_vi": {"type": "string"},
    },
    "required": ["status", "claims"],
}


def parse(text: str) -> Plan:
    """Raises ValueError (incl. pydantic.ValidationError) with a short, repair-friendly message."""
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    try:
        return Plan.model_validate_json(t)
    except ValidationError as e:
        raise ValueError("; ".join(f"{'.'.join(map(str, x['loc']))}: {x['msg']}" for x in e.errors()[:5]))


SYSTEM = """Bạn là trợ lý đọc Báo cáo thường niên 2025 của Techcombank. Chỉ trả về MỘT đối tượng JSON theo schema, không thêm chữ nào khác.
Quy tắc:
1. Chỉ dùng các đoạn bằng chứng [E1], [E2]... được cung cấp. Không dùng kiến thức bên ngoài. Mỗi claim phải liệt kê id bằng chứng hỗ trợ trong "evidence". Phần [chương › tiêu đề trang] ở đầu mỗi đoạn là tiêu đề của trang chứa đoạn đó: dùng nó để biết số liệu thuộc khối kinh doanh, công ty con hay báo cáo nào.
2. Số liệu trong text_vi phải được chép NGUYÊN VĂN từ bằng chứng (giữ định dạng Việt Nam, ví dụ 1.192.344.137 hoặc 14,6%), kèm đơn vị và kỳ (năm/ngày).
3. KHÔNG tự tính toán. Nếu cần tính (tăng trưởng, chênh lệch, tỷ lệ, tỷ trọng, CAGR, tổng, đổi đơn vị), khai báo trong "calculations" với inputs là id đại lượng [Q1], [Q2]... và chèn {C1}, {C2}... vào text_vi tại vị trí kết quả; đồng thời ghi cid vào "calc_refs". Nếu số cần tính nằm trong đoạn bằng chứng nhưng không có trong danh sách đại lượng, khai báo nó trong "literals" (lid "L1", evidence là id đoạn chứa số, number chép NGUYÊN VĂN, unit, period là năm hoặc ngày của số đó) rồi dùng "L1" làm input.
4. KHÔNG ghi số trang hay "[tr. N]" trong text_vi; hệ thống tự gắn trích dẫn.
5. Câu hỏi nhiều phần: liệt kê các phần trong "sub_questions" và trả lời từng phần bằng các claim riêng.
6. status: "answered" (đủ thông tin), "partial" (chỉ trả lời được một phần; ghi phần báo cáo không có vào clarification_vi), "not_in_report" (báo cáo không có thông tin; claims để trống), "ambiguous" (câu hỏi mơ hồ; điền clarification_vi, claims để trống).
7. Nếu bằng chứng không trả lời được câu hỏi thì chọn "not_in_report"; không đoán.
8. Viết tiếng Việt, ngắn gọn, đúng trọng tâm."""


def user_prompt(question: str, evidence: list[tuple[str, str]], quantities: list[str], history: str = "") -> str:
    ev = "\n\n".join(f"[{eid}]\n{txt}" for eid, txt in evidence)
    q = "\n".join(quantities) if quantities else "(không có)"
    h = f"Ngữ cảnh hội thoại trước đó (chỉ để hiểu câu hỏi, không phải bằng chứng):\n{history}\n\n" if history else ""
    return f"{h}BẰNG CHỨNG:\n{ev}\n\nĐẠI LƯỢNG CÓ THỂ DÙNG CHO PHÉP TÍNH:\n{q}\n\nCÂU HỎI: {question}\n\nTrả về JSON."

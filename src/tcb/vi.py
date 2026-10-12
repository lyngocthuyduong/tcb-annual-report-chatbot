"""Vietnamese text helpers: NFC, number canonicalisation, BM25 tokenizer (tcb-vi-v1)."""
from __future__ import annotations

import re
import unicodedata

# 1.192.344.137 | 1.192 | 14,6 | 40,4% | (181.150) | 2018–2025 | 31/12/2025 | B05/TCTD-HN | tr. 5
_TOK = re.compile(
    r"\d{1,2}/\d{1,2}/\d{4}"                      # dates
    r"|[a-z]\d{2}[a-z]?/[a-z\-]+"                 # form codes (text is lower-cased)
    r"|\d+(?:[.,]\d+)*%?"                        # numbers, kept whole
    r"|[^\W\d_]+(?:/[^\W\d_]+)?",                # words (incl. A/B abbreviations like N/N)
    re.UNICODE,
)
_VN_NUM = re.compile(r"^\d{1,3}(?:\.\d{3})+(?:,\d+)?%?$|^\d+,\d+%?$")


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def canon_num(tok: str) -> str:
    """Vietnamese number -> canonical form used as an extra BM25 token: 1.192.344.137 -> 1192344137, 14,6% -> 14.6%."""
    if _VN_NUM.match(tok):
        t = tok.rstrip("%")
        pct = "%" if tok.endswith("%") else ""
        if "," in t:
            t = t.replace(".", "").replace(",", ".").rstrip("0").rstrip(".")      # 16,0 == 16 ; 14,60 == 14,6
        else: t = t.replace(".", "")
        return t + pct
    return tok


def tokens(text: str) -> list[str]:
    """Lower-cased syllables + whole numbers (raw and canonical). Diacritics are kept; a folded copy is also emitted."""
    out: list[str] = []
    for m in _TOK.finditer(nfc(text).lower()):
        t = m.group(0)
        out.append(t)
        if t[0].isdigit():
            c = canon_num(t)
            if c != t: out.append(c)
        else:
            f = fold(t)
            if f != t: out.append(f)
    return out


def fold(s: str) -> str:
    s = unicodedata.normalize("NFD", s.replace("đ", "d").replace("Đ", "D"))
    return "".join(c for c in s if unicodedata.category(c) != "Mn")

"""UNIT: doctor + portability guards."""
import re
import sys
from pathlib import Path

import pytest

from tcb import conf, doctor

pytestmark = pytest.mark.unit


def test_pins_match_pyproject():
    toml = (Path(__file__).resolve().parent.parent / "pyproject.toml").read_text(encoding="utf-8")
    pins = dict(re.findall(r'"([A-Za-z0-9_.\-]+)==([0-9][^"]*)"', toml))
    assert pins == doctor.PINS


def test_doctor_reports_missing_pieces(tmp_path, monkeypatch):
    monkeypatch.setattr(conf, "MODELS_DIR", tmp_path / "models")
    monkeypatch.delenv("INDEX_DIR", raising=False)                     # compose sets an absolute INDEX_DIR=/app/idx
    rows = {r["check"]: r for r in doctor.checks(tmp_path)}
    assert rows["data/ar.pdf"]["status"] == "FAIL" and "git pull" in rows["data/ar.pdf"]["fix"]
    assert rows["model bge-m3"]["status"] == "FAIL" and "tcb.models download" in rows["model bge-m3"]["fix"]
    assert rows["index"]["status"] == "WARN"
    assert doctor.run(False, str(tmp_path)) == 1                       # any FAIL -> non-zero


def test_doctor_accepts_correct_model_snapshot(tmp_path, monkeypatch):
    m = tmp_path / "models"
    for sub, rev in (("bge-m3", conf.EMBED_REV),):
        (m / sub).mkdir(parents=True); (m / sub / ".rev").write_text(rev, encoding="utf-8"); (m / sub / "model.safetensors").write_bytes(b"x")
    monkeypatch.setattr(conf, "MODELS_DIR", m)
    rows = {r["check"]: r for r in doctor.checks(tmp_path)}
    assert rows["model bge-m3"]["status"] == "OK" and not any("reranker" in k for k in rows)
    (m / "bge-m3" / ".rev").write_text("deadbeef", encoding="utf-8")
    assert {r["check"]: r for r in doctor.checks(tmp_path)}["model bge-m3"]["status"] == "FAIL"


def test_real_pdf_hash_is_the_expected_one():
    pdf = Path(__file__).resolve().parent.parent / "data" / "ar.pdf"
    if not pdf.exists(): pytest.skip("no pdf")
    from tcb.build import sha256
    assert sha256(pdf) == conf.PDF_SHA256


def test_models_dir_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("TCB_MODELS_DIR", str(tmp_path / "m"))
    assert conf._models_dir() == tmp_path / "m"


def test_poppler_missing_gives_install_hint(monkeypatch):
    monkeypatch.setenv("PATH", ""); monkeypatch.delenv("POPPLER_BIN", raising=False)
    with pytest.raises(RuntimeError, match="poppler"): conf.poppler("pdftotext")


def test_cli_imports_without_unix_only_modules(monkeypatch):
    """Windows regression: `import resource` at module top crashed every command."""
    import importlib
    monkeypatch.setitem(sys.modules, "resource", None)           # makes `import resource` raise ImportError
    import tcb.cli as cli
    importlib.reload(cli)
    assert cli.peak_rss_mb() is None or isinstance(cli.peak_rss_mb(), int)


def test_load_env_parses_and_never_overrides(tmp_path, monkeypatch):
    p = tmp_path / ".env"
    p.write_text('﻿PROVIDER=ollama   # offline\nOPENAI_API_KEY="sk-abc"\n# c\nEMPTY=\nLLM_TIMEOUT_S=90\n', encoding="utf-8")
    for k in ("PROVIDER", "OPENAI_API_KEY", "EMPTY", "LLM_TIMEOUT_S"): monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("LLM_TIMEOUT_S", "5")                          # real env wins
    keys = conf.load_env(p)
    import os
    assert os.environ["PROVIDER"] == "ollama" and os.environ["OPENAI_API_KEY"] == "sk-abc"
    assert os.environ["LLM_TIMEOUT_S"] == "5" and "EMPTY" not in os.environ and "LLM_TIMEOUT_S" not in keys
    monkeypatch.delenv("PROVIDER"); monkeypatch.delenv("OPENAI_API_KEY")
    assert conf.load_env(tmp_path / "nope") == []

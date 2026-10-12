"""`tcb doctor`: one command that says what is missing on THIS machine and how to fix it. Read-only; never downloads."""
from __future__ import annotations

import importlib.metadata as md
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from . import conf

# keep in sync with pyproject.toml (tests/test_doctor.py enforces it)
PINS = {"llama-index-core": "0.14.25", "llama-index-embeddings-huggingface": "0.8.0", "sentence-transformers": "6.1.0",
        "transformers": "5.19.0", "torch": "2.14.1", "bm25s": "0.3.13", "fastapi": "0.143.0", "uvicorn": "0.54.0",
        "openai": "3.28.0", "anthropic": "1.13.0", "httpx": "0.28.1"}

OK, WARN, FAIL = "OK", "WARN", "FAIL"


def _ram_gb() -> float | None:
    try:
        if hasattr(os, "sysconf") and "SC_PHYS_PAGES" in os.sysconf_names:
            return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
        import ctypes

        class MS(ctypes.Structure):
            _fields_ = [("l", ctypes.c_ulong), ("m", ctypes.c_ulong), ("tp", ctypes.c_ulonglong), ("ap", ctypes.c_ulonglong),
                        ("tpf", ctypes.c_ulonglong), ("apf", ctypes.c_ulonglong), ("tv", ctypes.c_ulonglong), ("av", ctypes.c_ulonglong),
                        ("ae", ctypes.c_ulonglong)]
        s = MS(); s.l = ctypes.sizeof(MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))     # type: ignore[attr-defined]
        return s.tp / 1e9
    except Exception:
        return None


def checks(root: Path = Path(".")) -> list[dict]:
    out: list[dict] = []

    def add(name, status, detail, fix=""):
        out.append({"check": name, "status": status, "detail": detail, "fix": fix})

    v = sys.version_info
    add("python", OK if v >= (3, 12) else FAIL, f"{v.major}.{v.minor}.{v.micro} ({sys.executable})",
        "" if v >= (3, 12) else "Need Python >= 3.12. Create the venv with it: py -3.12 -m venv .venv (Windows) / python3.12 -m venv .venv")
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    add("virtualenv", OK if in_venv else WARN, "active" if in_venv else "not in a venv", "" if in_venv else "Activate .venv (see setup script) so pip installs do not touch the system Python")

    bad = []
    for pkg, want in PINS.items():
        try:
            have = md.version(pkg)
            if have != want: bad.append(f"{pkg} {have} (pinned {want})")
        except md.PackageNotFoundError:
            bad.append(f"{pkg} MISSING")
    add("packages", OK if not bad else FAIL, "all pinned versions installed" if not bad else "; ".join(bad[:6]),
        "" if not bad else 'pip install -e ".[dev]"   (install torch first: pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu)')

    for t in ("pdftotext", "pdfimages"):
        try:
            p = conf.poppler(t)
            ver = subprocess.run([p, "-v"], capture_output=True, text=True, encoding="utf-8", errors="replace")
            add(t, OK, (ver.stderr or ver.stdout).splitlines()[0] if (ver.stderr or ver.stdout) else p)
        except RuntimeError:
            add(t, FAIL, "not found", conf.POPPLER_HINT)

    pdf = root / "data" / "ar.pdf"
    if not pdf.exists():
        add("data/ar.pdf", FAIL, "missing", "git pull (the PDF is committed) or copy the report to data/ar.pdf")
    else:
        from .build import sha256
        h = sha256(pdf)
        add("data/ar.pdf", OK if h == conf.PDF_SHA256 else FAIL, f"sha256 {h[:12]}…",
            "" if h == conf.PDF_SHA256 else f"Different file than the one the code was built for (expected {conf.PDF_SHA256[:12]}…)")

    for sub, rev in (("bge-m3", conf.EMBED_REV),):
        d = conf.MODELS_DIR / sub
        r = d / ".rev"
        if not d.exists():
            add(f"model {sub}", FAIL, f"{d} missing", "python -m tcb.models download && python -m tcb.models convert   (set TCB_MODELS_DIR to change the folder)")
        elif not r.exists() or r.read_text(encoding="utf-8").strip() != rev:
            add(f"model {sub}", FAIL, f"{d} present but .rev != pinned {rev[:7]}", "delete the folder and download again")
        else:
            w = [p.name for p in d.iterdir() if p.suffix in (".safetensors", ".bin")]
            if sub == "bge-m3" and "model.safetensors" not in w:
                add(f"model {sub}", WARN, f"revision ok, weights {w}", "python -m tcb.models convert   (bge-m3 ships .bin only)")
            else:
                add(f"model {sub}", OK, f"{d} @ {rev[:7]}")

    idx = root / conf.settings().index_dir
    if (idx / "manifest.json").exists():
        from .index import IndexError_, validate
        try:
            m = validate(idx)
            add("index", OK, f"{idx}: {m['counts']['nodes']} nodes, {m['embedding']['model']}")
        except IndexError_ as e:
            add("index", FAIL, str(e).splitlines()[0], "rebuild: tcb build data/ar.pdf out/idx, then replace idx/")
    else:
        add("index", WARN, f"{idx} not built yet", "tcb build data/ar.pdf out/idx   (needs the real models)")

    env = root / ".env"
    prov = (os.getenv("PROVIDER") or "openai").lower()
    key = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY", "ollama": None}.get(prov, "?")
    have_key = key is None or bool(os.getenv(key)) or (env.exists() and any(
        l.startswith(f"{key}=") and l.split("=", 1)[1].strip() for l in env.read_text(encoding="utf-8").splitlines()))
    add("llm config", OK if have_key else WARN, f"PROVIDER={prov}" + ("" if have_key else f", {key} not set"),
        "" if have_key else "cp .env.example .env and fill the key (both Docker and local `tcb` read ./.env)")

    ram = _ram_gb()
    add("RAM", OK if ram is None or ram >= 6 else WARN, "unknown" if ram is None else f"{ram:.1f} GB", "" if ram is None or ram >= 6 else "BGE-M3 + index need ≈4 GB (peak RSS 4.1 GB measured); close other apps / raise Docker memory")
    free = shutil.disk_usage(root).free / 1e9
    add("disk", OK if free >= 15 else WARN, f"{free:.0f} GB free", "" if free >= 15 else "models ≈6 GB + image ≈10 GB")

    dk = shutil.which("docker")
    if not dk:
        add("docker", WARN, "not installed (only needed for the Docker path)", "https://docs.docker.com/get-docker/")
    else:
        try:
            r = subprocess.run([dk, "info"], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=20)
            add("docker", OK if r.returncode == 0 else WARN, "daemon reachable" if r.returncode == 0 else "installed but daemon not running",
                "" if r.returncode == 0 else "Start Docker Desktop and wait until it says 'running'")
        except Exception as e:
            add("docker", WARN, f"docker info failed: {e}", "")
    return out


def run(as_json: bool = False, root: str = ".") -> int:
    rows = checks(Path(root))
    if as_json:
        print(json.dumps(rows, ensure_ascii=False, indent=1))
    else:
        for r in rows:
            print(f"[{r['status']:<4}] {r['check']:<22} {r['detail']}")
            if r["fix"] and r["status"] != OK: print(f"        fix: {r['fix']}")
        n = {s: sum(r["status"] == s for r in rows) for s in (OK, WARN, FAIL)}
        print(f"\n{n[OK]} ok · {n[WARN]} warn · {n[FAIL]} fail")
    return 1 if any(r["status"] == FAIL for r in rows) else 0

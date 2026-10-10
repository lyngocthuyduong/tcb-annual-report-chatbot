#!/usr/bin/env bash
# Local (no Docker) setup for macOS / Linux / WSL / Git-Bash. Idempotent. Ends with `tcb doctor`.
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3.12}"; command -v "$PY" >/dev/null || PY=python3
"$PY" - <<'PYV'
import sys
if sys.version_info < (3, 12): sys.exit("Python >= 3.12 required (set PYTHON=/path/to/python3.12)")
PYV
[ -d .venv ] || "$PY" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate 2>/dev/null || source .venv/Scripts/activate
python -m pip install -q --upgrade pip
python -m pip install -q "torch==2.14.1" --index-url https://download.pytorch.org/whl/cpu
python -m pip install -q -e ".[dev]"
[ -f .env ] || cp .env.example .env
python -m tcb.cli models download      # ~6 GB, network, one-off
python -m tcb.cli models convert
python -m tcb.cli doctor
echo "next: python -m tcb.cli check && python -m tcb.cli build data/ar.pdf out/idx"

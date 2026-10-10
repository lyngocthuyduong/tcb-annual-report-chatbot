# Local (no Docker) setup for Windows PowerShell. Run:  powershell -ExecutionPolicy Bypass -File setup.ps1
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
function Run-Py { if (Get-Command py -ErrorAction SilentlyContinue) { py -3.12 @args } else { python @args } }
Run-Py -c "import sys; sys.exit(0 if sys.version_info >= (3,12) else 'Python >= 3.12 required (install from python.org)')"
if (-not (Test-Path .venv)) { Run-Py -m venv .venv }
. .\.venv\Scripts\Activate.ps1
python -m pip install -q --upgrade pip
python -m pip install -q "torch==2.14.1" --index-url https://download.pytorch.org/whl/cpu
python -m pip install -q -e ".[dev]"
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
if (-not (Get-Command pdftotext -ErrorAction SilentlyContinue) -and -not $env:POPPLER_BIN) {
  Write-Host "poppler missing: conda install -c conda-forge poppler  (or download poppler-windows, then set POPPLER_BIN to its bin folder)" -ForegroundColor Yellow
}
python -m tcb.cli models download
python -m tcb.cli models convert
python -m tcb.cli doctor
Write-Host "next: python -m tcb.cli check ; python -m tcb.cli build data/ar.pdf out/idx"

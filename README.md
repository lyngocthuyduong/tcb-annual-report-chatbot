# TCB — Vietnamese Chatbot over the Techcombank Annual Report 2025

Evidence-first: every answer cites printed pages `[tr. N]` taken from index metadata (never from the LLM); numbers in answers must appear in the cited evidence or come from a deterministic calculator; the bot refuses when the report lacks the answer. Design and measurements: `DESIGN.md`. Submission summary: `SUBMISSION.md`.

## Run (online)
```bash
cp .env.example .env          # set PROVIDER + its API key
docker compose up --build     # http://127.0.0.1:8000  (UI)  ·  /health  ·  POST /chat  ·  /docs
```
Needs `idx/` (shipped index) and `data/ar.pdf`. If `idx/` is missing, corrupt, or built with another model the server exits with code 3 and says how to fix it; it never rebuilds or downloads at run time. Missing key / bad `PROVIDER` → exit 2.

## Run (offline LLM, separate service, never an automatic fallback)
```bash
docker compose --profile offline up --build app-offline           # local Ollama + qwen2.5:7b (pulled once, ~4.7 GB)
docker compose --profile offline run --rm app-offline tcb smoke   # logs: "Ollama smoke test: Executed|Not Executed ..."
```

## Batch / eval
```bash
docker compose run --rm app tcb batch data/q10.json --out out/runs/q10       # answers + full traces, no human
docker compose run --rm app tcb eval  data/val.json --label val2             # scores, stage attribution, latency by stage
docker compose run --rm app tcb eval  --compare out/runs/eval/val1 out/runs/eval/val2
docker compose run --rm app tcb ask "Tỷ lệ CASA năm 2025?" --json
```
`eval` writes `traces.jsonl`, `results.jsonl` (score + failing stage per item), `summary.json` (pass rate, recall by stage, refusal P/R, p50/p95 latency, median seconds per stage, tokens, cost when the model has a price in `conf.PRICES`) and `config.json`. Only `"verified": true` items count.

## Run on Colab (development; no Docker there)
Runtime: **T4 GPU** only to rebuild the index (cell 4); everything else runs on the CPU (`TCB_DEVICE` defaults to `cpu`), which is also what latency numbers must use. Secrets (key icon, **Notebook access** ON): `gh-token`, `hf_token`, `anthropic_key` (or `openai_key`).
```python
# CELL 1: once per VM, then Runtime > Restart session
!apt-get -qq update && apt-get -qq install -y poppler-utils
import os; from google.colab import userdata
os.environ["GH_TOKEN"] = userdata.get("gh-token")
!git clone -q -b cont https://x-access-token:$GH_TOKEN@github.com/lyngocthuyduong/tcb-annual-report-chatbot /content/tcb
%cd /content/tcb
!pip install -q -e ".[dev]"
!pip uninstall -y -q torchvision torchaudio     # Colab's torchvision is built for another torch
```
```python
# CELL 2: after every restart (keeps files and packages, loses cwd and env vars)
import os; from google.colab import drive, userdata
drive.mount("/content/drive")
%cd /content/tcb
!git pull -q origin cont
!mkdir -p /content/drive/MyDrive/tcb-out && rm -rf out && ln -s /content/drive/MyDrive/tcb-out out
os.environ["TCB_MODELS_DIR"] = "/content/models"
for env, name in [("HF_TOKEN", "hf_token"), ("ANTHROPIC_API_KEY", "anthropic_key"), ("OPENAI_API_KEY", "openai_key")]:
    try: os.environ[env] = userdata.get(name)
    except Exception: print("no secret:", name)
os.environ.update(PROVIDER="anthropic", ANTHROPIC_MODEL="claude-haiku-5-5")
```
```python
# CELL 3: once per VM: BGE-M3 (≈2.3 GB) + gate
!python -m tcb.models download 2>&1 | tail -2
!tcb check | tee out/check.json            # dim 1024, revision ok, query_embed_s; if the .bin is refused: !python -m tcb.models convert
```
```python
# CELL 4: only when ingestion code changed (index format check fails): rebuild idx/ on the GPU, validate, commit
os.environ.update(JAX_PLATFORMS="cpu", XLA_PYTHON_CLIENT_PREALLOCATE="false")   # keep Colab's JAX off the T4
!rm -rf idx.new && TCB_DEVICE=cuda tcb build data/ar.pdf idx.new && tcb validate idx.new --snapshot && rm -rf idx && mv idx.new idx
!git config user.name "Ly Quoc Bien" && git config user.email "bien.anh2603@gmail.com"
!git add idx && git commit -qm "Rebuild index" && git push -q origin HEAD:cont
```
Then, one cell each, a NEW `--label`/`--out` every run:
```python
!python -m pytest -q                                              # unit + integ, no models
!tcb vset check data/dev.json && tcb vset check data/val.json     # gold facts / calc inputs are on their gold pages
!tcb exp data/dev.json data/val.json --label m5 2>&1 | tail -40   # retrieval only, no tokens
!tcb eval data/val.json --label val2
!tcb batch data/q10.json --out out/runs/q10-b
```
Offline mode on Colab: install Ollama (`curl -fsSL https://ollama.com/install.sh | sh`), start `ollama serve`, `ollama pull qwen2.5:7b`, set `PROVIDER=ollama OLLAMA_BASE_URL=http://localhost:11434 LLM_TIMEOUT_S=300`, then `tcb smoke`. Colab runs are not Docker proof; latency there is 2 vCPU.

| Symptom | Fix |
|---|---|
| `SecretNotFoundError` / `NotebookAccessError` | secret name differs, or Notebook access is off |
| `No module named tcb` or `/content/tcb` missing | the VM was recycled: cells 1 → restart → 2 → 3 |
| `torchvision::nms does not exist` | `pip uninstall torchvision torchaudio`, restart |
| `out` empty / dangling | Drive not mounted: re-run cell 2 |

## Tests: what each category proves
| Command | Proves | Does not prove |
|---|---|---|
| `pytest -m unit` | calculator, citations, refusal, retry, error mapping, API routing, scorers | any real model or provider works |
| `pytest -m integ` | page mapping, extraction, golden figures, index build/validate (hash embedder in tmp), pipeline wiring with a scripted LLM | retrieval quality |
| `pytest -m live_models` | BGE-M3 loads, dim and revision | answer quality |
| `pytest -m "live_openai or live_ollama"` | one schema-constrained call returns a valid, grounded plan | quality at scale |
| `pytest -m live` | shipped `idx/` + BGE-M3 + `PROVIDER` answer and refuse | |
| `./smoke.sh [offline]` | Docker build, `tcb check`, validate, tests, HTTP answer/refusal | quality |
| `tcb eval` | metrics on verified gold | |

Default `pytest` runs unit + integ. Skips are not passes. The hash embedder lives only in `tests/`; the production validator rejects any index not built with the pinned BGE-M3 revision.

## Layout
`src/tcb/`: `pdf` `parse` `nodes` `vi` (ingestion) · `build` `index` (index) · `find` (BM25 + dense → RRF → top 10) · `num` `calc` (quantities, calculator) · `plan` `gen` `llm/` (LLM plan, one repair retry, providers) · `cite` `refuse` (validation, policy) · `chat` `ask` (pipeline) · `api` `ui.html` `cli` `smoke` `doctor` · `ev` `exp` `vset` (evaluation) · `conf` (fixed model and constants) · `models` (Docker download).
`data/`: `ar.pdf`; `q10.json` (the 10 published questions, never tuned on); `dev.json` / `val.json` (70 / 34 items written by hand, checked against the PDF text by Claude, human spot-check pending).
Dependencies: exact pins in `pyproject.toml`; `requirements.lock` (hash-pinned, py3.12 linux-x86_64, no torch/CUDA) from `./lock.sh`; Docker installs CPU torch first.

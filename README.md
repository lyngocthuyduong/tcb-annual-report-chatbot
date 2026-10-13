# TCB — Vietnamese Chatbot over the Techcombank Annual Report 2025

Evidence-first: every answer cites printed pages `[tr. N]` taken from index metadata (never from the LLM); numbers in answers must appear in the cited evidence or come from a deterministic calculator; the bot refuses when the report lacks the answer. Read in this order: run (below) → batch → tests → `SUBMISSION.md` (decisions and results) → `DESIGN.md` (details and experiment log).

## Run (online)
```bash
mkdir -p out && cp .env.example .env   # set the API key (PROVIDER=anthropic by default); out/ first so the container can write to it
docker compose up --build     # http://127.0.0.1:8000  (UI)  ·  /health  ·  POST /chat  ·  /docs
```
Needs `idx/` (shipped index) and `data/ar.pdf`. If `idx/` is missing, corrupt, or built with another model the server exits with code 3 and says how to fix it; it never rebuilds or downloads at run time. Missing key / bad `PROVIDER` → exit 2.

If `/chat` returns 503 `APIConnectionError` while `/health` is fine, the container cannot reach the provider. Check DNS first (`docker compose exec app python -c "import socket; print(socket.gethostbyname('api.anthropic.com'))"`), then proxy, TLS and outbound access. In GitHub Codespaces DNS failed on the Compose network; a host-specific override fixed it without changing `compose.yaml`:
```bash
printf 'services:\n  app:\n    network_mode: bridge\n    dns: [1.1.1.1]\n' > /tmp/bridge.yaml
COMPOSE_FILE=compose.yaml:/tmp/bridge.yaml docker compose up -d app      # or: COMPOSE_FILE=... ./smoke.sh
```

## Run with a local model (Ollama on the host; not used for the reported results)
```bash
ollama pull qwen2.5:7b && OLLAMA_HOST=0.0.0.0 ollama serve       # on the host; containers reach it at host.docker.internal:11434
# .env: PROVIDER=ollama  LLM_TIMEOUT_S=300
docker compose up --build
docker compose run --rm app tcb smoke                             # prints "Ollama smoke test: Executed|Not Executed ..."
```
There is never an automatic fallback from one provider to another.

## Batch / eval
```bash
docker compose run --rm app tcb batch data/q10.json --out out/runs/q10       # any question file placed in data/; answers + full traces in out/
docker compose run --rm app tcb eval  data/val.json --label val2             # scores, stage attribution, latency by stage
docker compose run --rm app tcb eval  --compare out/runs/eval/val1 out/runs/eval/val2
docker compose run --rm app tcb eval  --rescore runs/test2                     # score saved answers again (no model call)
docker compose run --rm app tcb ask "Tỷ lệ CASA năm 2025?" --json
```
`eval` writes `traces.jsonl`, `results.jsonl` (score + failing stage per item), `summary.json` (pass rate, recall by stage, refusal P/R, p50/p95 latency, median seconds per stage, tokens, cost when the model has a price in `conf.PRICES`) and `config.json`. Only `"verified": true` items count.

## Tests: what each category proves
| Command | Proves | Does not prove |
|---|---|---|
| `pytest -m unit` | calculator, citations, refusal, retry, error mapping, API routing, scorers | any real model or provider works |
| `pytest -m integ` | page mapping, extraction, golden figures, index build/validate (hash embedder in tmp), pipeline wiring with a scripted LLM | retrieval quality |
| `pytest -m live_models` | BGE-M3 loads, dim and revision | answer quality |
| `pytest -m "live_anthropic or live_openai or live_ollama"` | one schema-constrained call returns a valid, grounded plan | quality at scale |
| `pytest -m live` | shipped `idx/` + BGE-M3 + `PROVIDER` answer and refuse | |
| `./smoke.sh` | Docker build, `tcb check`, validate, tests, HTTP answer/refusal | quality |
| `tcb eval` | metrics on verified gold | |

Default `pytest` runs unit + integ. Skips are not passes. The hash embedder lives only in `tests/`; the production validator rejects any index not built with the pinned BGE-M3 revision.

## Layout
`src/tcb/`: `pdf` `parse` `nodes` `vi` (ingestion) · `build` `index` (index) · `find` (BM25 + dense → RRF → top 12) · `num` `calc` (quantities, calculator) · `plan` `gen` `llm/` (LLM plan, one repair retry, providers) · `cite` `refuse` (validation, policy) · `chat` `ask` (pipeline) · `api` `ui.html` `cli` `smoke` `doctor` · `ev` `exp` `vset` (evaluation) · `conf` (fixed model and constants) · `models` (Docker download).
`data/`: `ar.pdf`; `q10.json` (the 10 published questions; used to debug, so not an independent benchmark); `dev.json` / `val.json` (70 / 34 items: development and validation); `test.json` (22 items written after the last code change, on pages no other set uses, run once as test1; its failures drove three fixes); `test2.json` (22 items written after those fixes, on pages no other set uses, run once). Gold facts machine-checked against the PDF text (`tcb vset check`) and reviewed by the author. `runs/`: summaries, configs, per-item results and traces of the reported runs.
Dependencies: exact pins in `pyproject.toml`; `requirements.lock` (hash-pinned, py3.12 linux-x86_64, no torch/CUDA) from `./lock.sh`; Docker installs CPU torch first.

## Run on Colab (development; two notebooks)
Secrets (key icon, **Notebook access** ON): `gh-token` (Contents read), `hf_token`, `anthropic_key`.

**Notebook A: build the index** (runtime T4 GPU; only when ingestion code changes; `tcb serve` refuses an index built by older code). Output: `idx.zip` (downloaded and copied to Drive); it is unzipped into the repo as `idx/` and committed from there.
```python
# A1: then Runtime > Restart session
!apt-get -qq update && apt-get -qq install -y poppler-utils
import os; from google.colab import userdata
os.environ["GH_TOKEN"] = userdata.get("gh-token")
!git clone -q -b cont https://x-access-token:$GH_TOKEN@github.com/lyngocthuyduong/tcb-annual-report-chatbot /content/tcb
%cd /content/tcb
!pip install -q -e ".[dev]" && pip uninstall -y -q torchvision torchaudio
```
```python
# A2: test, build, validate, zip; then unzip idx.zip into the repo as idx/ and commit it
import os; from google.colab import drive, files, userdata
drive.mount("/content/drive")
%cd /content/tcb
!git pull -q origin cont && git log --oneline -1
os.environ.update(TCB_MODELS_DIR="/content/models", HF_TOKEN=userdata.get("hf_token"),
                  JAX_PLATFORMS="cpu", XLA_PYTHON_CLIENT_PREALLOCATE="false")
!python -m tcb.models download 2>&1 | tail -1
!python -m pytest -q 2>&1 | tail -1                    # must pass before building
!rm -rf /content/build && mkdir -p /content/build && TCB_DEVICE=cuda tcb build data/ar.pdf /content/build/idx
!tcb validate /content/build/idx --snapshot && cat /content/build/idx/report.json
!cd /content/build && zip -rq idx.zip idx && cp idx.zip /content/drive/MyDrive/idx.zip && ls -la idx.zip
files.download("/content/build/idx.zip")
```

**Notebook B: run and measure** (runtime CPU: latency must come from the CPU)
```python
# B1: then Runtime > Restart session
!apt-get -qq update && apt-get -qq install -y poppler-utils
import os; from google.colab import userdata
os.environ["GH_TOKEN"] = userdata.get("gh-token")
!git clone -q -b cont https://x-access-token:$GH_TOKEN@github.com/lyngocthuyduong/tcb-annual-report-chatbot /content/tcb
%cd /content/tcb
!pip install -q -e ".[dev]" && pip uninstall -y -q torchvision torchaudio
```
```python
# B2: after every restart
import os; from google.colab import drive, userdata
drive.mount("/content/drive")
%cd /content/tcb
!git pull -q origin cont && git log --oneline -1
!mkdir -p /content/drive/MyDrive/tcb-out && rm -rf out && ln -s /content/drive/MyDrive/tcb-out out
os.environ.update(TCB_MODELS_DIR="/content/models", HF_TOKEN=userdata.get("hf_token"),
                  ANTHROPIC_API_KEY=userdata.get("anthropic_key"), PROVIDER="anthropic", ANTHROPIC_MODEL="claude-haiku-5-5")
```
```python
# B3: once per VM: model, gate, index check, tests
!python -m tcb.models download 2>&1 | tail -1
!tcb check
!tcb validate idx --snapshot
!python -m pytest -q 2>&1 | tail -1
```
```python
# B4: retrieval experiments (no tokens)
!tcb exp data/dev.json data/val.json --label m8 2>&1 | tail -40
```
```python
# B5: answer quality, latency, cost (a new --label each run). Index unchanged since 5c5f329: no rebuild needed
!tcb eval data/dev.json --label dev14
!tcb eval data/val.json --label val14
!for l in q10l q10m q10n; do tcb eval data/q10.json --label $l; done          # three samples: the Anthropic adapter uses the API default sampling
!for l in test2b test2c; do tcb eval data/test2.json --label $l; done         # test2 again at the submitted commit (first run: test2)
!cd out/runs/eval && zip -rq /content/runs.zip dev14 val14 q10l q10m q10n test2b test2c
from google.colab import files; files.download("/content/runs.zip")
```

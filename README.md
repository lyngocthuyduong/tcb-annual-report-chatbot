# TCB: Vietnamese chatbot over the Techcombank Annual Report 2025

Evidence-first: every answer cites printed pages `[tr. N]` taken from index metadata (never from the LLM); numbers in answers must appear in the cited evidence or come from a deterministic calculator; the bot refuses when the report lacks the answer.

**Start with [`SUBMISSION.md`](SUBMISSION.md)** (decisions, results, cost and latency). [`DESIGN.md`](DESIGN.md) holds the details and the experiment log.

Two answers recorded on the independent test set (`runs/test2`):

> **Hỏi:** Theo thuyết minh báo cáo tài chính hợp nhất, thu nhập từ dịch vụ thanh toán và tiền mặt năm 2025 tăng hay giảm bao nhiêu phần trăm so với năm 2024?
> **Đáp:** Theo thuyết minh báo cáo tài chính hợp nhất, thu nhập từ dịch vụ thanh toán và tiền mặt năm 2025 là 6.761.581 triệu đồng, giảm 16,7% so với năm 2024 (8.118.354 triệu đồng) [tr. 356].

The 16,7% comes from the Decimal calculator over the two statement cells; the model only chose the inputs.

> **Hỏi:** Techcombank đang nắm giữ bao nhiêu Dogecoin?
> **Đáp:** Báo cáo thường niên 2025 của Techcombank không có thông tin để trả lời câu hỏi này.

## Run
```bash
mkdir -p out && cp .env.example .env   # set the API key (PROVIDER=anthropic by default); out/ first so the container can write to it
docker compose up --build     # http://127.0.0.1:8000  (UI)  ·  /health  ·  POST /chat  ·  /docs
```
Needs `idx/` (shipped index) and `data/ar.pdf`. If `idx/` is missing, corrupt, or built with another model the server exits with code 3 and says how to fix it; it never rebuilds or downloads at run time. Missing key / bad `PROVIDER` → exit 2.

### Local model (Ollama on the host)
```bash
ollama pull qwen2.5:7b && OLLAMA_HOST=0.0.0.0 ollama serve       # on the host; containers reach it at host.docker.internal:11434
# .env: PROVIDER=ollama  LLM_TIMEOUT_S=300
docker compose up --build
docker compose run --rm app tcb smoke                             # one fixed question through the local model
```
I planned to report a local run as well. On the machines I had (2 vCPU, no GPU) a 7B model needs minutes per answer at these prompt sizes, so the reported results use claude-haiku-5-5; the Ollama adapter is tested against a mocked server. There is never an automatic fallback from one provider to another.

## Batch and evaluation
```bash
docker compose run --rm app tcb batch data/q10.json --out out/runs/q10       # any question file placed in data/; answers + full traces in out/
docker compose run --rm app tcb eval  data/val.json --label val2             # scores, stage attribution, latency by stage
docker compose run --rm app tcb eval  --compare out/runs/eval/val1 out/runs/eval/val2
docker compose run --rm app tcb eval  --rescore runs/test2                     # score saved answers again (no model call)
docker compose run --rm app tcb ask "Tỷ lệ CASA năm 2025?" --json
```
`eval` writes `traces.jsonl`, `results.jsonl` (score + failing stage per item), `summary.json` (pass rate, recall by stage, refusal P/R, p50/p95 latency, median seconds per stage, tokens, cost when the model has a price in `conf.PRICES`) and `config.json`. Only `"verified": true` items count.

## Repository map
- `src/tcb/`: `pdf` `parse` `nodes` `vi` (ingestion) · `build` `index` (index) · `find` (BM25 + dense → RRF → top 12) · `num` `calc` (quantities, calculator) · `plan` `gen` `llm/` (LLM plan, one repair retry, providers) · `cite` `refuse` (validation, policy) · `chat` `ask` (pipeline) · `api` `ui.html` `cli` `smoke` `doctor` · `ev` `exp` `vset` (evaluation) · `conf` (fixed model and constants) · `models` (Docker download).
- `data/`: `ar.pdf`; `q10.json` (the 10 published questions; used to debug, so not an independent benchmark); `dev.json` / `val.json` (70 / 34 items: development and validation); `test.json` (22 items written after the last code change, on pages no other set uses, run once as test1; its failures drove three fixes); `test2.json` (22 items written after those fixes, on pages no other set uses, run once). Gold facts machine-checked against the PDF text (`tcb vset check`) and reviewed by the author.
- `runs/`: the final runs with full traces (`dev15`, `val15`, `q10o` to `q10q`, `test2d`, `test2e`), the earlier test-set runs (`test1`, `test2`, rescored `test2r`), the retrieval experiment `exp/m7` and the Docker smoke summary; every other run cited in the docs is in `runs/history/` (summaries only).
- `idx/`: the shipped index, validated at start-up.
- Dependencies: exact pins in `pyproject.toml`; `requirements.lock` (hash-pinned, py3.12 linux-x86_64, no torch/CUDA) from `./lock.sh`; Docker installs CPU torch first.

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

Default `pytest` runs the unit and integration tests; live tests run on demand. The hash embedder lives only in `tests/`; the production validator rejects any index not built with the pinned BGE-M3 revision.

## Troubleshooting
If `/chat` returns 503 `APIConnectionError` while `/health` is fine, the container cannot reach the provider. Check DNS first (`docker compose exec app python -c "import socket; print(socket.gethostbyname('api.anthropic.com'))"`), then proxy, TLS and outbound access. In GitHub Codespaces DNS failed on the Compose network; a host-specific override fixed it without changing `compose.yaml`:
```bash
printf 'services:\n  app:\n    network_mode: bridge\n    dns: [1.1.1.1]\n' > /tmp/bridge.yaml
COMPOSE_FILE=compose.yaml:/tmp/bridge.yaml docker compose up -d app      # or: COMPOSE_FILE=... ./smoke.sh
```

## Development runbook
<details>
<summary>Colab notebooks used to build the index (T4 GPU) and to measure the reported runs (CPU)</summary>

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
# B5: answer quality, latency, cost (a new --label each run). The index has not changed since 5c5f329: no rebuild
!tcb eval data/dev.json --label dev15
!tcb eval data/val.json --label val15
!for l in q10o q10p q10q; do tcb eval data/q10.json --label $l; done          # three samples (the Anthropic API default sampling)
!for l in test2d test2e; do tcb eval data/test2.json --label $l; done         # test2 at the submitted commit (first run: test2)
!cd out/runs/eval && zip -rq /content/runs.zip dev15 val15 q10o q10p q10q test2d test2e
from google.colab import files; files.download("/content/runs.zip")
```

</details>

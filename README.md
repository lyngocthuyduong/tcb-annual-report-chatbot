# TCB — Vietnamese Chatbot over the Techcombank Annual Report 2025

Evidence-first: every answer cites printed pages `[tr. N]` taken from index metadata (never from the LLM); numbers in answers must appear in the cited evidence or come from a deterministic calculator; the bot refuses when the report lacks the answer.

## Status (read this first)
| Part | State | Evidence |
|---|---|---|
| PDF → page map → nodes → index build/validate (hash test embedder) | **verified** | `pytest -m integ` on the real PDF |
| Calculator, citation check, refusal, providers (mocked), repair retry, API, eval runner | implemented, **unit/mock-tested only** | `pytest -m unit` |
| Dependency lock | resolved (`requirements.lock`, `uv.lock`) — **never installed in Docker** | `lock.sh` |
| Real BGE-M3 + reranker load (`tcb check`) | **NOT RUN** | HF/PyTorch unreachable in the authoring sandbox |
| Real submission index `idx/` | **NOT BUILT** | needs the real BGE-M3 |
| Docker image / `docker compose up` / `smoke.sh` | **NOT BUILT / NOT RUN** | no Docker daemon in the sandbox; compose files only syntax-validated |
| OpenAI / Anthropic live call, Ollama smoke | **NOT RUN** | `tests/test_live.py` (opt-in, skips without keys/Ollama) |
| Experiments E1–E8, τ_low (E7), dev/val sets, quality numbers | **NOT RUN** | need the real index + human-verified gold |

## Run (online)
```bash
cp .env.example .env          # set PROVIDER + its API key
docker compose up --build     # http://127.0.0.1:8000  (UI)  ·  /health  ·  POST /chat  ·  /docs
```
Needs `idx/` (shipped index) and `data/ar.pdf`. If `idx/` is missing, corrupt, or built with other models the server exits with code 3 and says how to fix it; it never rebuilds or downloads at run time. Missing key / bad `PROVIDER` → exit 2.

## Run (offline LLM, separate service — never an automatic fallback)
```bash
docker compose --profile offline up --build app-offline   # local Ollama + qwen2.5:7b (pulled once, ~4.7 GB)
docker compose --profile offline run --rm app-offline tcb smoke   # logs: "Ollama smoke test: Executed|Not Executed ..."
```

## Batch / eval
```bash
docker compose run --rm app tcb batch data/q10.json --out out/runs/q10       # answers + full traces, no human
docker compose run --rm app tcb eval  data/q10.json --label first            # scores + stage attribution → out/runs/first/
docker compose run --rm app tcb ask "Tỷ lệ CASA năm 2025?" --json
```
`eval` writes `traces.jsonl` (full per-question trace), `results.jsonl` (per-item score + verdict), `summary.json`, `config.json` (provider, model, index, `submission_grade`). Items count only if `"verified": true` in the dataset (see `src/tcb/ev.py` for the schema); `data/seed.json` holds 4 **unverified** drafts for a pipeline check (`include_unverified`).

## Build the index (one-off, networked machine) — not done yet
```bash
docker compose build && docker compose run --rm app tcb check          # must pass first (real models)
mkdir -p out && chmod a+w out
docker compose run --rm app tcb build data/ar.pdf out/idx && rm -rf idx && mv out/idx idx
docker compose run --rm app tcb validate /app/idx --snapshot           # then commit idx/ (~43 MB)
```

## Tests — what each category does and does NOT prove
| Command | Category | Proves | Does not prove |
|---|---|---|---|
| `pytest -m unit` | unit / mock | logic: calculator, citations, refusal, retry, error mapping, API routing, scorers | any real model or provider works |
| `pytest -m integ` | integration, real parsed PDF | page mapping, extraction, golden figures, index build/validate (hash embedder in tmp) | retrieval quality; the submission index |
| `pytest -m live_models` | real models | BGE-M3/reranker load, dims, revisions (inside the image) | answer quality |
| `pytest -m "live_openai or live_ollama"` | real provider | one schema-constrained call returns a valid, grounded plan | quality at scale |
| `pytest -m live` | end-to-end | shipped `idx/` + real models + `PROVIDER` answer & refuse | — |
| `./smoke.sh [offline]` | Docker build/smoke | build, `tcb check`, validate, tests, HTTP answer/refusal | quality (that is `tcb eval`) |
| `tcb eval` | quality eval | metrics on verified gold | — |

Default `pytest` runs unit + integ only. Skips (live) are not passes. The hash embedder lives only in `tests/` and writes to tmp; the production validator rejects any index not built with the pinned BGE-M3 revision.

## Dependencies
`pyproject.toml` has exact pins; `requirements.lock` (hash-pinned, py3.12 linux-x86_64, no torch/CUDA) and `uv.lock` come from `./lock.sh`. The Dockerfile installs CPU torch from the PyTorch index, then `pip install --require-hashes -r requirements.lock`.

## Layout
`src/tcb/`: `pdf` `parse` `nodes` `vi` (ingestion) · `build` `index` (index) · `find` (BM25+dense→RRF→rerank) · `num` `calc` (quantities, calculator) · `plan` `gen` `llm/` (LLM plan, one repair retry, providers) · `cite` `refuse` (validation, policy) · `chat` `ask` (pipeline) · `api` `ui.html` `cli` `smoke` `ev` · `conf` (fixed models/constants) · `models` (Docker download).
`data/`: `ar.pdf`, `q10.json` (published questions; never tuned on), `seed.json` (unverified drafts).

## Known gaps
τ_low is unset (`conf.TAU_LOW=None`): the score-triggered retrieval retry is disabled until E7 runs on the real index. Low-quality FS tables (70 of 111 table chunks) are kept as text but yield no calculator quantities. Number grounding ignores integers ≤ 12.

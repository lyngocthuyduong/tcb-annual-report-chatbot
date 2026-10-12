<!-- Keep the headings. Items marked NOT RUN have not been executed; no number below is a measured result unless stated. -->

# Submission

## How to run
- Clean machine: `cp .env.example .env` (set `PROVIDER` + key), then `docker compose up --build` → http://127.0.0.1:8000. Prerequisites: Docker Compose v2, ~8 GB RAM for Docker, ~10 GB disk, internet for the first build. **Status: the Docker path is not yet built or run** (see README status table); the pipeline itself ran end-to-end outside Docker on Colab.
- Different key or local model: edit `.env` (`PROVIDER=openai|anthropic|ollama`, `OPENAI_API_KEY`/`ANTHROPIC_API_KEY`, `*_MODEL`). Offline LLM: `docker compose --profile offline up --build app-offline` (Ollama + `qwen2.5:7b`). Modes are separate services; there is no automatic fallback.
- Batch run: `docker compose run --rm app tcb batch data/q10.json --out out/runs/q10` (answers + traces); scoring: `tcb eval data/q10.json`.
- Environment variables: `PROVIDER`, `OPENAI_API_KEY`, `OPENAI_MODEL`, `ANTHROPIC_API_KEY`, `ANTHROPIC_MODEL`, `OLLAMA_BASE_URL`, `OLLAMA_MODEL`, `LLM_TIMEOUT_S`, `LLM_MAX_OUTPUT_TOKENS`, `INDEX_DIR`, `APP_HOST`, `APP_PORT`, `TRACE_DIR`, `LOG_LEVEL` (see `.env.example`). Embedding model, reranker, tokenizer, chunking are fixed in code and recorded in the index manifest.
- Shipped index: `idx/` in this repository (≈43 MB, committed): 1,526 nodes embedded with BAAI/bge-m3 (pinned revision) on a Colab T4 GPU; `tcb validate idx --snapshot` passes against the pinned models.

## Depth track(s) chosen
Document Intelligence + Retrieval + Evaluation. The report is a 196-page spread PDF with KPI tiles, financial-statement tables and a glossary; most failures in a naive pipeline come from page mapping, tables and Vietnamese numbers, so those get the engineering depth and the eval is built to attribute each miss to a stage.

## Architecture
PDF → (poppler bbox) half-page split of each two-page spread with printed-page recovery (footer, or inferred for dividers) → KPI-tile grouping, coordinate-based FS table rows (units, notes, nil/negative cells), glossary entries → typed nodes with provenance → LlamaIndex dense index (BGE-M3) + bm25s over a Vietnamese-number-aware tokenizer → manifest with sha256 per file, validated fail-closed at start-up. Query: optional follow-up rewrite → glossary expansion → BM25 top-40 + dense top-40 → RRF (k=60) → BGE-reranker-v2-m3 → 8 evidence blocks + extracted quantities → the LLM returns a JSON plan (claims with evidence ids, calculator calls) → deterministic validation (ids, no page numbers in text, every number must appear in the cited evidence) → calculator (Decimal, unit/period checks) → refusal policy → answer with `[tr. N]` from node metadata. One repair retry on the same provider; provider errors are surfaced, never swapped.

## What I tried that did not work
- planned pdfplumber layout extraction replaced by `pdftotext -bbox-layout` (reliable word boxes on the spread layout; parsed with a regex because the output is not well-formed XML).
- bm25s default tokenizer: splits `1.192.344.137` and `14,6%`; replaced with `tcb-vi-v1` (numbers kept whole + canonical forms + folded diacritics).
- First table-column anchoring mixed absolute and offset coordinates and lost the first column; fixed and covered by golden tests (B02 row "Tiền mặt, vàng": note 5, 4.360.821 / 3.384.770).
- Complex note tables produce scrambled headers (47 of 107 table chunks); they are flagged `quality=low`, kept as text, and excluded from calculator quantities rather than guessed. The income statement and cash flow were initially in this group (year header row read as data, units merged into labels); fixed and covered by golden tests, so B01–B04 now yield quantities.
- A Docker build on a 32 GB GitHub Codespace failed twice with `no space left on device` while exporting layers (models were copied into extra layers by `chown -R` and a safetensors convert). Fixed in the Dockerfile (`COPY --chown`, no convert); **not re-run**. The index is built once on Colab instead (README).
- Real models and the index were first exercised on Colab (not the authoring sandbox). Docker, OpenAI and Ollama paths are still untested (below).

## Evaluation
- Method: deterministic scorers (fact presence incl. unit alternatives, citation validity, page hit vs gold/alt pages, refusal P/R, stage attribution). No LLM judge. Items count only when human-verified.
- Results on the 10 published questions: **NOT RUN** (needs the real index and a provider). What is verified: all 9 answerable gold facts are present, as text, on the gold printed pages of the parsed document (`tests/test_reg.py`) — extraction coverage, not answer quality.
- What the numbers made you change: nothing yet (no quality numbers exist).

## Cost and latency
- Ingestion (Colab, 2 CPU cores + T4 GPU for embedding): ingest 8.6 s, embed 81.5 s (GPU), BM25 0.6 s, 1,526 nodes (`idx/report.json`). CPU-only embedding time **NOT MEASURED**.
- Model load on Colab CPU (`tcb check`): BGE-M3 21.9 s, reranker 2.2 s, peak RSS 4,145 MB (disk cache warm; not Docker).
- Per query: one observed question with claude-haiku-5-5: LLM 2.0 s, 4,647 in / 327 out tokens; total 83 s including first model load (CPU). Steady-state latency **NOT MEASURED**. Traces record tokens, calls, latency and a list-price cost estimate per question (gpt-4o-mini $0.15/$0.60 per 1M tokens; claude-sonnet-5-5 $2/$10; Ollama $0).
- Models used: BAAI/bge-m3 (embedding), BAAI/bge-reranker-v2-m3 (rerank); generation: gpt-4o-mini by default, claude-sonnet-5-5 or Ollama `qwen2.5:7b` selectable.

## What I deliberately did not build
Another orchestration framework (LlamaIndex is used for ingestion/retrieval only), an LLM judge, OCR for image-only content (inventoried, not read), automatic provider fallback, runtime index rebuild or model download.

## With 10x time and budget
1. Run the chunk-size experiment, and tune τ_low and the reranker pool on a larger set labelled by people (dev/val here were checked by Claude against the PDF text; human spot-check pending).
2. Recover the 47 low-quality note tables with a layout model and enable calculator quantities for them.
3. Judge model calibrated against ≥40 human labels; multi-document support.

## Demo video
NOT RECORDED.

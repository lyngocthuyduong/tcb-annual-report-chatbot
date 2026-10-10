<!-- Keep the headings. Items marked NOT RUN have not been executed; no number below is a measured result unless stated. -->

# Submission

## How to run
- Clean machine: `cp .env.example .env` (set `PROVIDER` + key), then `docker compose up --build` → http://127.0.0.1:8000. Prerequisites: Docker Compose v2, ~8 GB RAM for Docker, ~10 GB disk, internet for the first build. **Status: not yet built or run end-to-end** (see README status table).
- Different key or local model: edit `.env` (`PROVIDER=openai|anthropic|ollama`, `OPENAI_API_KEY`/`ANTHROPIC_API_KEY`, `*_MODEL`). Offline LLM: `docker compose --profile offline up --build app-offline` (Ollama + `qwen2.5:7b`). Modes are separate services; there is no automatic fallback.
- Batch run: `docker compose run --rm app tcb batch data/q10.json --out out/runs/q10` (answers + traces); scoring: `tcb eval data/q10.json`.
- Environment variables: `PROVIDER`, `OPENAI_API_KEY`, `OPENAI_MODEL`, `ANTHROPIC_API_KEY`, `ANTHROPIC_MODEL`, `OLLAMA_BASE_URL`, `OLLAMA_MODEL`, `LLM_TIMEOUT_S`, `LLM_MAX_OUTPUT_TOKENS`, `INDEX_DIR`, `APP_HOST`, `APP_PORT`, `TRACE_DIR`, `LOG_LEVEL` (see `.env.example`). Embedding model, reranker, tokenizer, chunking are fixed in code and recorded in the index manifest.
- Shipped index: `idx/` in this repository (≈43 MB, committed). **Not built yet** — it requires the real BGE-M3.

## Depth track(s) chosen
Document Intelligence + Retrieval + Evaluation. The report is a 196-page spread PDF with KPI tiles, financial-statement tables and a glossary; most failures in a naive pipeline come from page mapping, tables and Vietnamese numbers, so those get the engineering depth and the eval is built to attribute each miss to a stage.

## Architecture
PDF → (poppler bbox) half-page split of each two-page spread with printed-page recovery (footer, or inferred for dividers) → KPI-tile grouping, coordinate-based FS table rows (units, notes, nil/negative cells), glossary entries → typed nodes with provenance → LlamaIndex dense index (BGE-M3) + bm25s over a Vietnamese-number-aware tokenizer → manifest with sha256 per file, validated fail-closed at start-up. Query: optional follow-up rewrite → glossary expansion → BM25 top-40 + dense top-40 → RRF (k=60) → BGE-reranker-v2-m3 → 8 evidence blocks + extracted quantities → the LLM returns a JSON plan (claims with evidence ids, calculator calls) → deterministic validation (ids, no page numbers in text, every number must appear in the cited evidence) → calculator (Decimal, unit/period checks) → refusal policy → answer with `[tr. N]` from node metadata. One repair retry on the same provider; provider errors are surfaced, never swapped.

## What I tried that did not work
- planned pdfplumber layout extraction replaced by `pdftotext -bbox-layout` (reliable word boxes on the spread layout; parsed with a regex because the output is not well-formed XML).
- bm25s default tokenizer: splits `1.192.344.137` and `14,6%`; replaced with `tcb-vi-v1` (numbers kept whole + canonical forms + folded diacritics).
- First table-column anchoring mixed absolute and offset coordinates and lost the first column; fixed and covered by golden tests (B02 row "Tiền mặt, vàng": note 5, 4.360.821 / 3.384.770).
- Complex note tables produce scrambled headers; they are flagged `quality=low`, kept as text, and excluded from calculator quantities rather than guessed.
- Real BGE-M3 / PyTorch / Docker could not be reached from the authoring sandbox, so the real index, image and live-provider paths are untested (below).

## Evaluation
- Method: deterministic scorers (fact presence incl. unit alternatives, citation validity, page hit vs gold/alt pages, refusal P/R, stage attribution). No LLM judge. Items count only when human-verified.
- Results on the 10 published questions: **NOT RUN** (needs the real index and a provider). What is verified: all 9 answerable gold facts are present, as text, on the gold printed pages of the parsed document (`tests/test_reg.py`) — extraction coverage, not answer quality.
- What the numbers made you change: nothing yet (no quality numbers exist).

## Cost and latency
- Ingestion: parse + node build measured only with the hash embedder (seconds, CPU); real-embedding cost/time **NOT MEASURED**. 1,542 nodes to embed.
- Per query: **NOT MEASURED**. Traces record tokens, calls, latency and a list-price cost estimate per question (gpt-4o-mini $0.15/$0.60 per 1M tokens; claude-sonnet-5-5 $2/$10; Ollama $0).
- Models used: BAAI/bge-m3 (embedding), BAAI/bge-reranker-v2-m3 (rerank); generation: gpt-4o-mini by default, claude-sonnet-5-5 or Ollama `qwen2.5:7b` selectable.

## What I deliberately did not build
Another orchestration framework (LlamaIndex is used for ingestion/retrieval only), an LLM judge, OCR for image-only content (inventoried, not read), automatic provider fallback, runtime index rebuild or model download.

## With 10x time and budget
1. Build the real index, run E1–E8 (RRF k, chunk size, reranker pool) and freeze τ_low (E7) on a larger human-verified dev/val set.
2. Recover the 70 low-quality note tables with a layout model and enable calculator quantities for them.
3. Judge model calibrated against ≥40 human labels; multi-document support.

## Demo video
NOT RECORDED.

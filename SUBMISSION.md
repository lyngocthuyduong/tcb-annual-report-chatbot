<!-- Keep the headings. Items marked NOT RUN have not been executed; every number below is measured unless marked. Details: DESIGN.md. -->

# Submission

## How to run
- Clean machine: `cp .env.example .env` (set `PROVIDER` + key), then `docker compose up --build` → http://127.0.0.1:8000. Prerequisites: Docker Compose v2, ~6 GB RAM for Docker, ~6 GB disk (estimated), internet for the first build. **The Docker path is not yet built or run**; the pipeline ran end to end outside Docker on Colab.
- Different key or local model: edit `.env` (`PROVIDER=openai|anthropic|ollama`, `OPENAI_API_KEY` / `ANTHROPIC_API_KEY`, `*_MODEL`). Offline LLM: `docker compose --profile offline up --build app-offline` (Ollama + `qwen2.5:7b`). Modes are separate services; there is no automatic fallback.
- Batch run: `docker compose run --rm app tcb batch questions.json --out out/runs/x` (any JSON list of `{"id", "question"}`; answers + full traces); scoring: `tcb eval data/q10.json`.
- Environment variables: `PROVIDER`, `OPENAI_API_KEY`, `OPENAI_MODEL`, `ANTHROPIC_API_KEY`, `ANTHROPIC_MODEL`, `OLLAMA_BASE_URL`, `OLLAMA_MODEL`, `LLM_TIMEOUT_S`, `LLM_MAX_OUTPUT_TOKENS`, `INDEX_DIR`, `APP_HOST`, `APP_PORT`, `TRACE_DIR`, `LOG_LEVEL` (see `.env.example`). The embedding model, tokenizer, chunking and evidence count are fixed in code and recorded in the index manifest.
- Shipped index: `idx/` (≈ 43 MB, committed): 1,526 nodes embedded with BAAI/bge-m3 (pinned revision); `tcb validate idx --snapshot` passes.

## Depth track(s) chosen
Document Intelligence + Retrieval + Evaluation. In this report the failure points are the page mapping (two printed pages per PDF page), the KPI tiles and statement tables, and Vietnamese number formats, so the engineering depth goes there, and the eval attributes each miss to a stage so that each change is driven by a measured cause.

## Architecture
**Ingestion.** `pdftotext -bbox-layout` word boxes; each 1190 × 842 pt spread is split at x ≈ 595 into two printed pages, numbered from the footer (190 halves) or inferred for dividers, because the grader checks printed pages and a PDF page holds two. Content is typed by layout: KPI tiles (value + label + period grouped by coordinates), statement tables (rows rebuilt from column x-positions; units, note references, nil `-` and negative `(x)` cells; every row carries its column header and unit), narrative chunks (≤ 220 syllable-words, never crossing a page), glossary entries (one per term). Each node keeps doc id, PDF page, half, printed page, page source, section and type. Tables are flagged `high`/`low` quality; only the 60 `high` chunks (all of B01–B04 among them) yield calculator quantities.

**Index.** BGE-M3 dense vectors (LlamaIndex in-memory store) + bm25s over `tcb-vi-v1` tokens (numbers kept whole and in canonical form, diacritics also folded). A manifest pins the PDF hash, model revision, tokenizer, chunking and a sha256 per file; the server refuses to start on any mismatch.

**Query.** Follow-ups are rewritten into a standalone question (and checked to keep the user's numbers). Glossary abbreviations are expanded with the report's own definition. BM25 top 40 + dense top 40 → reciprocal rank fusion (k = 60) → the top 10 blocks go to the LLM with the table quantities that best match the question. There is no reranker (below).

**Generation contract.** The LLM returns only a JSON plan (forced tool call / JSON schema): claims, each naming evidence ids; calculations naming quantity ids `Q*` or literals `L*` (a number copied from an evidence block with unit and period); a status; a clarification. It never writes page numbers or does arithmetic.

**Deterministic checks.** Every number in a claim must occur in the evidence it cites (or the question); every literal must occur verbatim in its evidence block; the calculator (Decimal; growth, difference, ratio, share, CAGR, sum, unit conversion) refuses mixed units or periods; claims that fail are dropped after one repair attempt; `[tr. N]` is rendered from node metadata and re-checked against the retrieved pages. Status is `answered`, `partial`, `not_in_report`, `ambiguous` or `error`; provider errors name provider and model and are never swapped for another provider.

## What I tried that did not work
- **Cross-encoder reranking (BGE-reranker-v2-m3).** On dev+val (85 answerable items), reranking 20 to 40 fused candidates found a gold page for 0.894 of items, the plain fused top 10 for 0.882: one item. It cost about 2 s per candidate on 2 CPU cores (≈ 60 s per question at 30; p50 30.3 s on val with 10). With 10 candidates and 10 evidence blocks it would only reorder them, so it was removed (model, download, gate, experiments), which also takes ≈ 2.2 GB off the image.
- **Score-threshold retry (τ_low).** A threshold on the top rerank score flagged unanswerable questions (τ = 0.599), but the retry was a second full search on exactly those questions and recovered no page in any experiment; removed with the reranker.
- **bm25s default tokenizer.** It splits `1.192.344.137` into `192`, `344`, `137`. It still matched `tcb-vi-v1` by one item at R@20 (0.894 vs 0.882) but lost at R@5 (0.800 vs 0.824); kept `tcb-vi-v1`.
- **Calculator over table quantities only.** Val showed that any arithmetic on numbers in narrative text or KPI tiles (e.g. headcount 12.705 vs 11.848) was refused. Replaced by literals checked against the evidence text.
- **Speed ideas rejected after measuring:** a vector database (dense search over 1,526 × 1,024 vectors takes 49 ms in LlamaIndex, 0.3 ms in FAISS: not where the time goes); BGE-M3 sparse weights instead of BM25 (same forward pass, no time saved; only 1 of 85 questions contains the number it asks for, and "2025" is in 572 of 1,526 chunks); 400 to 500-word chunks with overlap (prompts ≈ 2.5× longer, and overlap across pages breaks one-page citations).
- **Extraction.** pdfplumber was replaced by poppler word boxes on this spread layout; the first table-column anchoring lost the first column (fixed, golden tests); 47 of 107 table chunks (complex notes) still have scrambled headers and are kept as text only.
- **Docker on a 32 GB Codespace** ran out of disk while exporting layers (weights copied by `chown -R` and a format conversion); the Dockerfile was fixed but **not re-run**.

## Evaluation
- Method: deterministic scorers, no LLM judge: fact presence with unit alternatives, citation validity, page hit against gold and alternative pages, refusal precision/recall, and the first failing stage (recall miss, fusion drop, rank drop, citation, calculation, generation, over/under-refusal). Retrieval experiments (`tcb exp`) cost no tokens. Dev 70 / val 34 items written by hand across statements, KPIs, narrative, segments, subsidiary, governance, glossary, calculations, follow-ups and 19 unanswerable or ambiguous questions; golds checked by Claude against the PDF text, human spot-check pending.
- Results on the 10 published questions: **NOT RUN as a scored eval.** A batch run on CPU (with the former reranker, pool 30) answered 7, gave 1 partial and refused 1 answerable question; the unanswerable one was refused correctly. Val (34 items, claude-haiku-5-5, with the former reranker): 31 / 34 pass, facts 0.893, citations valid 1.0, page hit 0.893, no false answer on the 5 unanswerable, 3 answerable refused.
- What the numbers made you change: the reranker and τ_low retry were removed (E4/E7 + latency); literals were added to the calculator and quantities are selected by relevance when over the cap (the 3 val refusals); the tokenizer and RRF k stayed (E2, E3b). Val was used for these fixes, so it is no longer held-out; q10 never drove a decision.

## Cost and latency
- Ingestion (Colab, 2 vCPU + T4 for embedding): parse 8.6 s, embed 81.5 s (GPU), BM25 0.6 s, 1,526 nodes. CPU-only embedding **NOT MEASURED**.
- Per query: val with the former reranker on 2 vCPU: p50 30.3 s, p95 35.5 s; about 20 s of that was reranking, now removed; **latency without the reranker NOT YET MEASURED**. One question with claude-haiku-5-5 used 4,647 input / 327 output tokens and 2.0 s of LLM time. `tcb eval` reports tokens, p50/p95 and median seconds per stage; the dollar cost is reported only for models with a list price in `conf.PRICES`.
- Models used: embedding BAAI/bge-m3; generation claude-haiku-5-5 for development runs, gpt-4o-mini / claude-sonnet-5-5 / Ollama `qwen2.5:7b` selectable. Which model produced each number is recorded in every run's `config.json`.

## What I deliberately did not build
An agent or orchestration framework (LlamaIndex is used for ingestion and vector retrieval only), an LLM judge, OCR or vision for image-only content (715 images inventoried, not read), automatic provider fallback, runtime index rebuild or model download, authentication, multi-user serving, deployment.

## With 10x time and budget
1. A larger eval set labelled by people, a fresh held-out split, and an LLM judge validated against those labels.
2. Recover the 47 low-quality note tables with a layout model so they yield quantities.
3. Benchmark other Vietnamese embedding models and a sparse/learned lexical model against BM25 on that set.
4. Multi-document support (per-document manifests and page maps).

## Demo video
NOT RECORDED.

# Submission

## How to run
- Clean machine: `cp .env.example .env` (set `PROVIDER` and its key), then `docker compose up --build` → http://127.0.0.1:8000 (web UI, `POST /chat`, `/docs`). Prerequisites: Docker Compose v2, ~6 GB RAM for Docker, internet for the first build.
- Different key or local model: edit `.env` (`PROVIDER=openai|anthropic|ollama`, `OPENAI_API_KEY` / `ANTHROPIC_API_KEY`, `*_MODEL`). Fully offline: `docker compose --profile offline up --build app-offline` (Ollama + `qwen2.5:7b`). Online and offline are separate services with no automatic fallback.
- Batch run: `docker compose run --rm app tcb batch questions.json --out out/runs/x`, where `questions.json` is a list of `{"id", "question"}`; writes answers and full traces. Scoring against gold: `tcb eval data/q10.json`.
- Environment variables: `PROVIDER`, `OPENAI_API_KEY`, `OPENAI_MODEL`, `ANTHROPIC_API_KEY`, `ANTHROPIC_MODEL`, `OLLAMA_BASE_URL`, `OLLAMA_MODEL`, `LLM_TIMEOUT_S`, `LLM_MAX_OUTPUT_TOKENS`, `INDEX_DIR`, `APP_HOST`, `APP_PORT`, `TRACE_DIR`, `LOG_LEVEL` (all in `.env.example`). The embedding model, tokenizer and chunking are fixed in code and recorded in the index manifest.
- Shipped index: `idx/` in this repository (≈ 41 MB).

## Depth track(s) chosen
**Document intelligence** and **Retrieval and reasoning**, with an evaluation harness that attributes every failure to a stage. The report is a two-page-spread PDF whose answers live in KPI tiles, audited statement tables, a glossary and narrative pages; the errors that matter to an analyst (wrong figure, wrong page) start in page mapping, table structure and Vietnamese number formats.

## Architecture
**Pages.** 197 PDF pages = cover + 196 spreads. Each spread is split at the gutter into two printed pages; 380 of 393 halves take their number from the footer, 13 dividers are inferred from their neighbours. The half-page is the citation unit, so `[tr. N]` always names the printed page.

**Typed nodes (1,512).** Word boxes from `pdftotext -bbox-layout` are grouped by layout:
- KPI tiles (180): label, figure, unit and change grouped by coordinates and ordered label → figure → unit, e.g. `Dư nợ vay | 328,1 | Nghìn tỷ đồng | 26,9% N/N`. Number-only fragments (chart ticks, contents-page numbers) are not indexed.
- Statement tables (115 chunks): rows rebuilt from column positions with note references, nil `-` and negative `(x)` cells; wrapped labels rejoined, sub-headings kept as their own rows, column headers reduced to period + unit; every row carries them. 60 chunks (all of B01 to B04, and 39 notes) have a period in every column and expose typed quantities; a continuation page inherits its statement's title.
- Narrative (1,054 chunks of ≤ 220 syllable-words) read column by column, and glossary entries (163 terms).
- Every node text starts with its context `[chapter › page title › subtitle]`, read from font size: a page title carries over to the following pages until the next one, so a tile on p.59 reads `[Thành tựu của Tech › Khối Ngân hàng Bán lẻ (RBG) › Điểm nhấn 2025] Dư nợ vay | 328,1 …`.

**Index.** BGE-M3 dense vectors (LlamaIndex) and BM25 over a Vietnamese tokenizer that keeps numbers whole (`1.192.344.137`, `14,6%`) and also indexes diacritic-folded forms. A manifest pins the PDF hash, model revision, tokenizer, chunking and a sha256 per file; the server refuses to start on any mismatch.

**Retrieval.** Follow-ups are rewritten into a standalone question. Abbreviations are expanded with the report's own glossary (`CASA` → `CASA (Tiền gửi không kỳ hạn)`). BM25 top 40 and dense top 40 are fused with reciprocal rank fusion (k = 60); the top 10 blocks go to the LLM with the table quantities whose labels best match the question.

**Generation contract.** The LLM returns only a JSON plan (forced tool call / JSON schema): claims, each naming its evidence ids, and calculations over quantity ids `Q*` (from tables) or literals `L*` (a number copied from an evidence block with its unit and period). The model never writes page numbers and never does arithmetic.

**Deterministic checks.** Every number in a claim must occur in its cited evidence; every literal must occur verbatim in the block it names; the calculator (Decimal: growth, difference, ratio, share, CAGR, sum, unit conversion) rejects mixed units and periods; failing claims trigger one repair call, then are dropped. Citations are rendered from node metadata and re-checked against the retrieved pages. Answers end as `answered`, `partial`, `not_in_report`, `ambiguous` or `error`; provider errors name the provider and model.

## What I tried that did not work
- **Cross-encoder reranking** (BGE-reranker-v2-m3, dev+val, 85 answerable items): reranking 20 to 40 fused candidates put a gold page in the final blocks for 0.894 of items, the fused top 10 for 0.882 (one item), at about 2 s of CPU per candidate (≈ 60 s per question for 30). Removed; the Docker image is 2.2 GB smaller.
- **Retry on a low rerank score (τ_low)**: the threshold that best separated unanswerable questions (0.599) would trigger a second full search on exactly those questions, and no experiment showed the retry recovering a page. Removed with the reranker.
- **bm25s default tokenizer**: splits `1.192.344.137` into `192`, `344`, `137`; BM25-only R@5 0.800 against 0.824 for the Vietnamese tokenizer.
- **Calculator limited to table quantities**: arithmetic on figures from narrative text or KPI tiles (e.g. headcount 12.705 vs 11.848, p.52) was refused. Replaced by literals checked against the evidence text.
- **Sorting text blocks by position** interleaved the two columns of prose wherever a heading spanned both (166 groups on 109 pages). Replaced by a column-aware reading order.
- **Chunk-level context only from the chapter name**: a KPI tile on the RBG spread (`Dư nợ vay 328,1`) did not say it belonged to the Retail block, so "RBG loan balance" was refused. Fixed with page-title context; a paragraph spanning two KPI tiles also glued them into one narrative chunk, fixed in the grouping rule (+7 tiles recovered).
- **Speed ideas measured and dropped**: a vector database (top-40 over 1,526 × 1,024 vectors: 49 ms in LlamaIndex, 0.3 ms in FAISS, against tens of seconds of reranking), BGE-M3 sparse weights in place of BM25 (computed in the same forward pass, so no time saved; 1 of 85 questions contains a figure other than a year or date), 400 to 500-word chunks with overlap (≈ 2.5× longer prompts; overlap across pages breaks one-page citations).

## Evaluation
- Method: deterministic scoring, no LLM judge: fact presence with unit alternatives, citation validity, page hit against gold and alternative pages, refusal precision and recall, and the first failing stage (recall miss, fusion drop, rank drop, citation, calculation, generation, over/under-refusal), printed per failed item from its trace. Retrieval experiments (`tcb exp`) cost no tokens. Dev (70) and val (34) are hand-written across statements, KPI tiles, narrative, business segments, subsidiaries, governance, investor relations, glossary, calculations, follow-ups and 19 unanswerable or ambiguous questions; each gold fact was checked against the PDF text of its page and each calculation recomputed.
- Results on the 10 published questions: <!-- fill from `tcb eval data/q10.json --label q10` -->
- What the numbers made you change: E4 and latency removed the reranker; E2 kept the Vietnamese tokenizer; E3b kept RRF k = 60; the three val failures led to literals in the calculator and quantity selection by question relevance; the batch refusal of sq-07 (RBG loan balance, p.59) led to page-title context in every node, a general fix tested on other pages (RBG, TCBS, TCC, board, cash-flow continuation, glossary).

## Cost and latency
- Ingestion (Colab, 2 vCPU + T4 for embedding): parse 13.4 s, embed 80.1 s for 1,512 nodes, BM25 0.7 s (`idx/report.json`). One-off, no API cost.
- Per query: <!-- fill from summary.json of the final val/q10 run: p50, p95, stage_seconds_p50, tokens, cost_usd_per_question; state CPU and model -->
- Models used: embedding BAAI/bge-m3; generation claude-haiku-5-5 (development), gpt-4o-mini, claude-sonnet-5-5 or Ollama `qwen2.5:7b` selectable; each run's `config.json` records the model that produced its numbers.

## What I deliberately did not build
An agent framework (LlamaIndex is used for ingestion and vector retrieval only), an LLM judge, OCR or vision for image-only content (428 images on 117 pages inventoried), automatic provider fallback, runtime index rebuild or model download, authentication, multi-user serving, deployment.

## With 10x time and budget
1. A larger eval set labelled by people with a fresh held-out split, and an LLM judge validated against those labels.
2. A layout model for the 47 note tables whose headers do not parse cleanly, so they expose quantities.
3. Benchmark other Vietnamese embedding models and learned sparse retrieval against BM25 on that set.
4. Multi-document support: one manifest and page map per document.

## Demo video
<!-- link -->

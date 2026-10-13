# Submission

## How to run
- Clean machine: `cp .env.example .env` (set `PROVIDER` and its key), then `docker compose up --build` → http://127.0.0.1:8000 (web UI, `POST /chat`, `/docs`). Prerequisites: Docker Compose v2, ~6 GB RAM for Docker, internet for the first build.
- Different key or local model: edit `.env` (`PROVIDER=openai|anthropic|ollama`, `OPENAI_API_KEY` / `ANTHROPIC_API_KEY`, `*_MODEL`). Fully offline: `docker compose --profile offline up --build app-offline` (Ollama + `qwen2.5:7b`). Online and offline are separate services with no automatic fallback.
- Batch run: `docker compose run --rm app tcb batch questions.json --out out/runs/x`, where the file is `data/sample_questions.json`-shaped (a list of `{"id", "question", ...}`), a list of strings, `{"questions": [...]}`, JSON Lines or plain text with one question per line; questions are answered independently and `answers.jsonl` holds each answer, status, citations and full trace. Scoring against gold: `tcb eval data/q10.json`.
- Environment variables: `PROVIDER`, `OPENAI_API_KEY`, `OPENAI_MODEL`, `ANTHROPIC_API_KEY`, `ANTHROPIC_MODEL`, `OLLAMA_BASE_URL`, `OLLAMA_MODEL`, `OLLAMA_NUM_CTX`, `LLM_TIMEOUT_S`, `LLM_MAX_OUTPUT_TOKENS`, `INDEX_DIR`, `APP_HOST`, `APP_PORT`, `TRACE_DIR`, `LOG_LEVEL` (all in `.env.example`). The embedding model, tokenizer and chunking are fixed in code and recorded in the index manifest.
- Shipped index: `idx/` in this repository (≈ 41 MB).

## Depth track(s) chosen
**Document intelligence** and **Retrieval and reasoning**, with an evaluation harness that attributes every failure to a stage. The report is a two-page-spread PDF whose answers live in KPI tiles, audited statement tables, a glossary and narrative pages; the errors that matter to an analyst (wrong figure, wrong page) start in page mapping, table structure and Vietnamese number formats.

## Architecture
**Pages.** 197 PDF pages = cover + 196 spreads. Each spread is split at the gutter into two printed pages; 380 of 393 halves take their number from the footer, 13 dividers are inferred from their neighbours. The half-page is the citation unit, so `[tr. N]` always names the printed page.

**Typed nodes (1,445).** Word boxes from `pdftotext -bbox-layout` are grouped by layout:
- KPI tiles (180): label, figure, unit and change grouped by coordinates and ordered label → figure → unit, e.g. `Dư nợ vay | 328,1 | Nghìn tỷ đồng | 26,9% N/N`. Number-only fragments (chart ticks, contents-page numbers) are not indexed.
- Statement tables (119 chunks): rows rebuilt from column positions with note references, nil `-` and negative `(x)` cells; wrapped labels rejoined, sub-headings kept as their own rows, column headers reduced to period + unit; every row carries them. A row of years inside a table is the header of the sub-table below it. Note tables are named by their numbered heading (`41. GIAO DỊCH VỚI CÁC BÊN LIÊN QUAN`), carried across pages. 66 chunks (all 16 of the primary statements and 50 from the notes) have a period in every column and expose typed quantities.
- Narrative (983 chunks of ≤ 220 syllable-words) read column by column; a sentence that wraps into the next column or around a KPI tile is rejoined before chunking. Glossary entries (163 terms). The running header and footer, and the statement band repeated on 80 pages (statement name, period line, form code `B05/TCTD-HN`), are removed as furniture; the form code is kept as table metadata.
- Images: 428 on 117 pages. A visual audit of every image covering ≥ 1 % of a half-page found no chart or table that exists only as pixels (backdrops, logos, portraits, photos, adverts); badges and diagrams carry vector text, which is read. No OCR or vision model.
- Every node text starts with its context `[chapter › page title › subtitle]`, read from font size: a page title carries over to the following pages until the next one, so a tile on p.59 reads `[Thành tựu của Tech › Khối Ngân hàng Bán lẻ (RBG) › Điểm nhấn 2025] Dư nợ vay | 328,1 …`.

**Index.** BGE-M3 dense vectors (LlamaIndex) and BM25 over a Vietnamese tokenizer that keeps numbers whole (`1.192.344.137`, `14,6%`) and also indexes diacritic-folded forms. A manifest pins the PDF hash, model revision, tokenizer, chunking and a sha256 per file; the server refuses to start on any mismatch.

**Retrieval.** Follow-ups are rewritten into a standalone question. Abbreviations are expanded with the report's own glossary (`CASA` → `CASA (Tiền gửi không kỳ hạn)`). BM25 top 40 and dense top 40 are fused with reciprocal rank fusion (k = 60); the top 20 blocks go to the LLM with the table quantities whose labels best match the question.

**Generation contract.** The LLM returns only a JSON plan (forced tool call / JSON schema): claims, each naming its evidence ids, and calculations over quantity ids `Q*` (from tables) or literals `L*` (a number copied from an evidence block with its unit and period). The model never writes page numbers and never does arithmetic. When two blocks give different figures for the same item and period (not a rounding or unit difference), the prompt tells it to report both, each with its own citation, rather than pick one.

**Deterministic checks.** Every number in a claim must occur in its cited evidence; every literal must occur verbatim in the block it names; a number must keep the magnitude its evidence prints (`1.192 nghìn tỷ đồng`, not `1.192 tỷ` or a bare `1.192`); the calculator (Decimal: growth, difference, ratio, share, CAGR, sum, unit conversion) rejects mixed units and periods; failing claims trigger one repair call, then are dropped. Citations are rendered from node metadata and re-checked against the retrieved pages. Answers end as `answered`, `partial`, `not_in_report`, `ambiguous` or `error`; provider errors name the provider and model.

**Where no language model is used.** Page mapping, table structure, number parsing, page citations, arithmetic, number and magnitude verification, and the final answer status are code. The LLM rewrites follow-ups, chooses evidence and words the answer.

**Specific to this document.** The spread split and footer page map, the title font sizes (35 / 27 pt), the statement band and form-code pattern (`B05/TCTD-HN`, the Vietnamese credit-institution forms) and the glossary page marker (`Thuật ngữ`). The rest (Vietnamese number and unit handling, column rebuilding from coordinates, retrieval, the plan contract and checks, the evaluation harness) carries to other Vietnamese reports; each new layout needs its page map and title thresholds checked against a few pages.

## What I tried that did not work
- **Cross-encoder reranking** (BGE-reranker-v2-m3, dev+val, 85 answerable items): reranking 20 to 40 fused candidates put a gold page in the final blocks for 0.894 of items, the fused top 10 for 0.882 (one item), at about 2 s of CPU per candidate (≈ 60 s per question for 30). Removed; the Docker image is 2.2 GB smaller.
- **Retry on a low rerank score (τ_low)**: the threshold that best separated unanswerable questions (0.599) would trigger a second full search on exactly those questions, and no experiment showed the retry recovering a page. Removed with the reranker.
- **bm25s default tokenizer**: splits `1.192.344.137` into `192`, `344`, `137`; BM25-only R@5 0.800 against 0.824 for the Vietnamese tokenizer.
- **Calculator limited to table quantities**: arithmetic on figures from narrative text or KPI tiles (e.g. headcount 12.705 vs 11.848, p.52) was refused. Replaced by literals checked against the evidence text.
- **Column headers read from the text above the numbers**: on the notes pages the form code `B05/TCTD-HN` (top right) became the header of the last column (p.369 showed `B05/TCTD-HN: 710.409` for the 2024 value), and a tax-rate sentence made a column `%` (p.351). Fixed by stopping at the form-code line, treating a row of years as the header of the sub-table below it, and taking the unit from the line under the years: 2 more note tables became machine-readable and one wrong unit was removed.
- **Sorting text blocks by position** interleaved the two columns of prose wherever a heading or a centred box spanned both (166 groups on 109 pages; on p.4 it split "302 chi nhánh và / phòng giao dịch tại 29 trên 34 tỉnh thành"). Replaced by a column-aware reading order; text chunks that start mid-sentence fell from 56 to 7.
- **Chunk-level context only from the chapter name**: a KPI tile on the RBG spread (`Dư nợ vay 328,1`) did not say it belonged to the Retail block, so "RBG loan balance" was refused. Fixed with page-title context; a paragraph spanning two KPI tiles also glued them into one narrative chunk, fixed in the grouping rule (+7 tiles recovered).
- **Speed ideas measured and dropped**: a vector database (top-40 over 1,526 × 1,024 vectors: 49 ms in LlamaIndex, 0.3 ms in FAISS, against tens of seconds of reranking), BGE-M3 sparse weights in place of BM25 (computed in the same forward pass, so no time saved; 1 of 85 questions contains a figure other than a year or date), 400 to 500-word chunks with overlap (≈ 2.5× longer prompts; overlap across pages breaks one-page citations).

## Evaluation
- Method: deterministic scoring, no LLM judge: fact presence with unit alternatives, citation validity, page hit against gold and alternative pages, refusal precision and recall, and the first failing stage (recall miss, fusion drop, rank drop, citation, calculation, generation, over/under-refusal), printed per failed item from its trace. Retrieval experiments (`tcb exp`) cost no tokens. Dev (70) and val (34) are hand-written across statements, KPI tiles, narrative, business segments, subsidiaries, governance, investor relations, glossary, calculations, follow-ups and 19 unanswerable or ambiguous questions; each gold fact was checked against the PDF text of its page and each calculation recomputed.
- Results on the 10 published questions: <!-- fill from `tcb eval data/q10.json --label q10` -->
- What the numbers made you change: E4 and latency removed the reranker and set the evidence count to 20 on the final index (fused recall 0.859 at 10, 0.906 at 15, 0.929 at 20); E2 kept the Vietnamese tokenizer; E3b kept RRF k = 60; the three val failures led to literals in the calculator and quantity selection by question relevance; the batch refusal of sq-07 (RBG loan balance, p.59) led to page-title context in every node, a general fix tested on other pages (RBG, TCBS, TCC, board, cash-flow continuation, glossary).

## Cost and latency
- Ingestion (Colab, 2 vCPU + T4 for embedding): parse 20.5 s, embed 82.4 s for 1,445 nodes, BM25 1.0 s (`idx/report.json`). One-off, no API cost.
- Per query: <!-- fill from summary.json of the final val/q10 run: p50, p95, stage_seconds_p50, tokens, cost_usd_per_question; state CPU and model -->
- Models used: embedding BAAI/bge-m3; generation claude-haiku-5-5 (development), gpt-4o-mini, claude-sonnet-5-5 or Ollama `qwen2.5:7b` selectable; each run's `config.json` records the model that produced its numbers.

## What I deliberately did not build
An agent framework (LlamaIndex is used for ingestion and vector retrieval only), an LLM judge, OCR or vision for image-only content (428 images on 117 pages inventoried), automatic provider fallback, runtime index rebuild or model download, authentication, multi-user serving, deployment.

## With 10x time and budget
1. A larger eval set labelled by people with a fresh held-out split, and an LLM judge validated against those labels.
2. A layout model for the 31 note tables whose headers do not parse cleanly, so they expose quantities.
3. Benchmark other Vietnamese embedding models and learned sparse retrieval against BM25 on that set.
4. Multi-document support: one manifest and page map per document.

## Demo video
<!-- link -->

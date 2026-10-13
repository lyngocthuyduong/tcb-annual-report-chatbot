# Submission

## How to run
- Clean machine: `cp .env.example .env` (set `PROVIDER` and its key), then `docker compose up --build` → http://127.0.0.1:8000 (web UI, `POST /chat`, `/docs`). Prerequisites: Docker Compose v2, about 6 GB of RAM for Docker, internet for the first build. The server validates the shipped index and the PDF before it accepts a question and exits with a named error (code 3) otherwise.
- Different key or local model: edit `.env` (`PROVIDER=openai|anthropic|ollama`, `OPENAI_API_KEY` / `ANTHROPIC_API_KEY`, `*_MODEL`). Local model: `docker compose --profile offline up --build app-offline` (Ollama + `qwen2.5:7b`); the first run pulls images, packages and the model, after which no external API is called. Online and offline are separate services; there is no silent fallback from one to the other.
- Batch run: place the question file in `data/` (mounted into the container) and run `docker compose run --rm app tcb batch data/questions.json --out out/runs/x`. Accepted formats: the brief's `sample_questions.json` shape, a list of strings, `{"questions": [...]}`, JSON Lines, or plain text with one question per line. `out/runs/x/answers.jsonl` holds each answer with its status, citations and full trace. Scoring against gold: `tcb eval <file>`.
- Environment variables: `PROVIDER`, `OPENAI_API_KEY`, `OPENAI_MODEL`, `ANTHROPIC_API_KEY`, `ANTHROPIC_MODEL`, `OLLAMA_BASE_URL`, `OLLAMA_MODEL`, `OLLAMA_NUM_CTX`, `LLM_TIMEOUT_S`, `LLM_MAX_OUTPUT_TOKENS`, `INDEX_DIR`, `APP_HOST`, `APP_PORT`, `TRACE_DIR`, `LOG_LEVEL` (all documented in `.env.example`). Embedding model, tokenizer and chunking are fixed in code and pinned in the index manifest, so a grader cannot run the shipped index with a mismatched configuration.
- Shipped index: `idx/` in this repository (39 MB).

## Depth track(s) chosen
**Document intelligence** and **Retrieval and reasoning**, backed by an evaluation harness that attributes every failure to a stage.

The two tracks address one risk from both ends. An analyst loses trust on a wrong figure or a wrong page, and in this report both originate before the model is ever called: in two-page spreads, multi-column prose, KPI tiles whose label, figure and unit are separate text boxes, statement tables whose headers span several lines, and Vietnamese number formats. Document intelligence makes the right block retrievable and machine-readable; retrieval and reasoning make sure the answer carries only what that block proves.

## Architecture
**Pages.** 197 PDF pages are a cover and 196 spreads. Each spread is split at the gutter; 380 of 393 half-pages take their number from the printed footer and 13 dividers are inferred from their neighbours. The half-page is the citation unit, so `[tr. N]` always names the page the analyst sees.

**Typed nodes (1,445).** Word boxes from `pdftotext -bbox-layout` are grouped by layout, not by text order:
- KPI tiles (180): label, figure, unit and change reassembled from coordinates, e.g. `Dư nợ vay | 328,1 | Nghìn tỷ đồng | 26,9% N/N`.
- Statement tables (119 chunks): rows rebuilt from column positions with note references, nil `-` and negative `(x)` cells; every row carries its column's period and unit; a row of years is read as the header of the sub-table below it; note tables are named by their numbered heading (`41. GIAO DỊCH VỚI CÁC BÊN LIÊN QUAN`) across page breaks. 66 chunks (all 16 primary-statement chunks and 50 note chunks) have a resolved period in every column and expose typed quantities to the calculator.
- Narrative (983 chunks, at most 220 syllable-words) read column by column, with sentences that wrap around a column break or a KPI tile rejoined before chunking. Glossary (163 terms).
- Every node starts with its context, read from font size and carried across pages: `[Thành tựu của Tech › Khối Ngân hàng Bán lẻ (RBG) › Điểm nhấn 2025] Dư nợ vay | 328,1 …`, so a tile knows which business it belongs to.
- Page furniture is removed: running headers and footers, and the statement band repeated on 80 pages (statement name, period line, form code); the form code is kept as table metadata.
- Images: 428 on 117 pages. A visual audit of every image covering at least 1 % of a half-page found no chart or table that exists only as pixels; badges and diagrams carry vector text, which is extracted. OCR is therefore not on the critical path for this report.

**Index.** BGE-M3 dense vectors (LlamaIndex) and BM25 over a Vietnamese tokenizer that keeps numbers whole (`1.192.344.137`, `14,6%`) and also indexes diacritic-folded forms. A manifest pins the PDF hash, model revision, tokenizer, chunking and a sha256 per file.

**Retrieval.** Follow-ups are rewritten into a standalone question (a rewrite that drops a number or introduces a name absent from the conversation is rejected). Abbreviations are expanded with the report's own glossary (`CASA` → `Tiền gửi không kỳ hạn`). BM25 top 40 and dense top 40 are fused with reciprocal rank fusion (k = 60); the top 12 blocks go to the model with the table quantities whose labels best match the question.

**Generation contract.** The model returns a JSON plan only: sentences, each naming the evidence blocks it rests on, and calculations over table quantities (`Q*`) or figures it copies from a block (`L*`, with unit and period). It never writes a page number and never computes. Sentences are joined into one paragraph in an investor-relations register: the figure, its change and the remark the report prints beside it, stated once, with every page that prints it.

**What each layer guarantees.** The layers are complementary: each closes a failure the others cannot see.

| Analyst risk | Layer | Guarantee |
|---|---|---|
| Wrong page | page map + metadata citations | every `[tr. N]` is the printed page of a retrieved block, inserted by code; a final pass rejects any page outside the retrieved evidence |
| Wrong figure, unit or scale | number grounding | every figure in a sentence is printed in its cited block with the same magnitude and sign (`1.192 nghìn tỷ đồng`, never `1.192 tỷ`); numbers in the question are premises to verify, not evidence |
| Wrong arithmetic | calculator | Decimal arithmetic (growth, difference, ratio, share, CAGR, sum, unit conversion) over typed quantities; mixed units, mismatched periods or a missing input stop the calculation instead of producing a number |
| An invented input to a calculation | literal verification | a copied figure is accepted only if its number, its printed unit and its year are found in the block it names; anything that cannot be proven is not computed, and the answer says what is missing |
| Answering what the report does not contain | refusal policy | the final status (`answered`, `partial`, `not_in_report`, `ambiguous`) is decided by code from the plan and the checks; a sentence that fails a check gets one repair call, then is dropped and the answer is marked partial |
| Silent drift between code, index and PDF | manifest validation | the server refuses to start if the PDF, model revision, tokenizer, chunking or any index file differs from what the index was built with |

The scope is deliberate: these layers bind every number, page and calculation, which is where an analyst's credibility is spent. The qualitative wording around them is bound to the retrieved blocks by the plan contract and measured in evaluation; a per-sentence entailment check is the next layer (see the 10x section).

**Where no language model is used.** Page mapping, table structure, number parsing, citations, arithmetic, verification and the answer status. The model is used where language is the task: rewriting follow-ups, choosing which retrieved block supports which sentence, and wording the answer.

**What carries to the next hundred documents.** Vietnamese number and unit handling, column rebuilding from coordinates, retrieval, the plan contract and every check, and the evaluation harness are document-agnostic. Per-layout configuration is small and explicit: the spread split and footer page map, two title font sizes, the statement band and form-code pattern of Vietnamese credit-institution statements, the glossary page marker and the issuer's name.

## What I tried that did not work
- **Cross-encoder reranking** (BGE-reranker-v2-m3, 85 answerable dev+val items): reranking 20 to 40 fused candidates found a gold page for 0.894 of items against 0.882 for the fused list alone, one item, at about 60 s of CPU per question. Removed, along with a low-score retry it enabled; p50 latency fell from 30 s to about 2 s and the image shrank by 2.2 GB.
- **The default BM25 tokenizer** splits `1.192.344.137` into `192`, `344`, `137` (BM25 R@5 0.800 against 0.824 with the Vietnamese tokenizer).
- **Text order from the PDF**: blocks sorted by position interleaved two-column prose wherever a heading or centred box spanned both columns (166 groups on 109 pages). A column-aware reading order cut chunks that start mid-sentence from 56 to 7.
- **Column headers taken from the text above the numbers**: on note pages the form code at the top right became the last column's header, and a tax-rate sentence made a column `%`. Headers now stop at the page band, rows of years become sub-table headers, and units come from the line under the years.
- **Context from the chapter name only**: a KPI tile on the Retail spread did not say it was Retail, so "RBG loan balance" was refused. Page-title context in every node fixed it across segments and subsidiaries.
- **Calculator restricted to table quantities**: refused arithmetic on figures that live in prose or tiles (headcount 12.705 vs 11.848). Replaced by literals verified against the block.
- **Removing sentences that restate a figure**: it also removed remarks they carried ("dẫn đầu hệ thống ngân hàng"). Restatement is now handled by the prompt and flagged in the trace.
- **Number grounding on raw digits**: comparing raw and canonical forms let `179,501` (179.5) match `179.501` (179 thousand). A red-team pass on the checks led to the current design: canonical comparison plus magnitude, sign, unit and period, each with an adversarial test.
- **Speed ideas measured and dropped**: a vector database (49 ms → 0.3 ms per search, irrelevant next to the model call), BGE-M3 sparse weights instead of BM25 (same forward pass, no saving), 400 to 500-word chunks with overlap (2.5× longer prompts, and overlap across pages breaks one-page citations).

## Evaluation
- Method: deterministic scoring, no LLM judge. Per item: facts present with their magnitude, citation validity, page hit against gold pages, refusal correctness, and the first failing stage (recall, fusion, rank, citation, calculation, generation, refusal) read from the trace. Retrieval experiments (`tcb exp`) cost no tokens.
- Four sets with separate roles. Every gold fact was machine-checked by Claude against the PDF text of its page and every calculation recomputed; answers marked below were also read by me.

| Set | Items | Role |
|---|---|---|
| `dev.json` | 70 | development: experiments and debugging |
| `val.json` | 34 | validation: configuration choices (evidence count, tokenizer, fusion) |
| `q10.json` | 10 | the published questions; also used for debugging |
| `test.json` | 22 | written after the last code change on pages no other set uses, frozen, run once: the independent estimate |

- Results (commit <!-- fill -->, claude-haiku-5-5, final index):
<!-- fill from runs/: q10 automatic + read by me; test automatic + read by me; val; dev -->
- What the numbers made you change: E4 and latency removed the reranker; the evidence count is set by a fixed rule (smallest k within 2 points of the best recall) and re-derived on each rebuilt index, giving k = 12 on the final one (fused recall 0.871 at 10, 0.929 at 12); E2 kept the Vietnamese tokenizer and E3b kept RRF k = 60; validation failures led to literals and to question-aware quantity selection; reading every answer led to the magnitude, sign and canonical-number checks and to the paragraph register. Run artifacts (summary, config with commit and hardware, per-item results and traces) are in `runs/`.

## Cost and latency
- Ingestion (Colab, 2 vCPU + T4 for embedding): parse 20.5 s, embed 82.4 s for 1,445 nodes, BM25 1.0 s (`idx/report.json`). One-off; no API cost.
- Per query (Colab, 2 vCPU, no GPU; claude-haiku-5-5 at $0.10 / $0.50 per million input / output tokens):
<!-- fill from runs/: p50, p95, tokens, cost per question, stage medians -->
  - One-off at start: BGE-M3 load 25.1 s, peak memory 2.5 GB.
- Models used: embedding BAAI/bge-m3; generation claude-haiku-5-5 for every reported number (gpt-4o-mini, claude-sonnet-5-5 or Ollama `qwen2.5:7b` selectable); each run's `config.json` records the model.

## What I deliberately did not build
- An LLM judge: deterministic scorers are free, reproducible and cannot be talked into a pass; a judge is only worth its cost once validated against human labels.
- OCR or vision: the image audit found no information that exists only as pixels in this report.
- Calculator access to the 31 multi-level note tables (provision matrices, maturity gaps): they are indexed and answerable as text, but a guessed column would put a wrong number into an analyst's email, so they expose no typed quantities until a layout model resolves their headers.
- An agent framework, automatic provider fallback, runtime index rebuild, authentication, multi-user serving and deployment, as the brief asks.

## With 10x time and budget
1. A per-sentence entailment check on qualitative wording, run on the cited block only (one extra small-model call per sentence), validated against human labels.
2. A larger evaluation set labelled by analysts, with a held-out split refreshed per release and an LLM judge calibrated on it.
3. A layout model for the multi-level note tables, so that every table exposes typed quantities.
4. A Vietnamese embedding benchmark (BGE-M3 against alternatives and learned sparse retrieval) on that set.
5. Multi-document support: one manifest and page map per report, and retrieval scoped by issuer and year.

## Demo video
<!-- link -->

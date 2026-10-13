# Submission

An analyst pastes the answer into an email, so they must be able to check it in seconds: the figure, its unit and the page it came from. Every design choice below serves that check. Figures, units, pages and arithmetic are produced or verified by code; the language model only chooses which retrieved evidence answers the question and words the answer.

## How to run
- Clean machine: `cp .env.example .env`, set `PROVIDER` and its key, then `docker compose up --build` → http://127.0.0.1:8000 (UI, `POST /chat`, `/docs`). Prerequisites: Docker Compose v2, about 6 GB of RAM for Docker, internet for the first build. Before serving, the app validates the shipped index and the PDF and exits with a named error (code 3) on any mismatch.
- Different key or local model: set `PROVIDER=openai|anthropic|ollama` and the matching key or model in `.env`. Local model: `docker compose --profile offline up --build app-offline` (Ollama + `qwen2.5:7b`; the first run downloads it). There is no automatic fallback between providers.
- Batch run: put the question file in `data/` and run `docker compose run --rm app tcb batch data/questions.json --out out/runs/x`; `answers.jsonl` holds each answer, status, citations and trace. Accepts the brief's `sample_questions.json` shape, a list of strings, JSON Lines or one question per line. Scoring: `tcb eval <file>`.
- Environment variables: `PROVIDER`, `OPENAI_API_KEY`, `OPENAI_MODEL`, `ANTHROPIC_API_KEY`, `ANTHROPIC_MODEL`, `OLLAMA_BASE_URL`, `OLLAMA_MODEL`, `OLLAMA_NUM_CTX`, `LLM_TIMEOUT_S`, `LLM_MAX_OUTPUT_TOKENS`, `INDEX_DIR`, `APP_HOST`, `APP_PORT`, `TRACE_DIR`, `LOG_LEVEL` (described in `.env.example`). Embedding model, tokenizer and chunking are fixed in code and pinned in the index manifest.
- Shipped index: `idx/` (39 MB).

## Depth track(s) chosen
**Document intelligence** and **Retrieval and reasoning.** In this report a wrong figure or page usually starts before the model is called: two-page spreads, multi-column prose, KPI tiles split into separate text boxes, statement headers spread over several lines, Vietnamese number formats. Document intelligence makes the right evidence findable and machine-readable; retrieval and reasoning keep the answer to what that evidence proves. An evaluation harness ties every failure to the stage that caused it.

## Architecture
Four layers, each responsible for one part of the check.

1. **Extraction keeps structure.** Each spread is split into printed pages (380 of 393 numbered from their footer, 13 inferred), so `[tr. N]` is the page the analyst sees. Layout, not text order, defines 1,445 typed nodes: KPI tiles reassembled as label, figure, unit and change, with the up or down triangle drawn beside a change kept as a marker (`7% (▲)`); statement rows rebuilt from column positions with each column's period and unit; prose read column by column; glossary entries. Each node carries its page title as context (`[… › Khối Ngân hàng Bán lẻ (RBG)] Dư nợ vay | 328,1 | Nghìn tỷ đồng`).
2. **Retrieval finds evidence.** BM25 over a Vietnamese tokenizer that keeps numbers whole, plus BGE-M3 dense vectors, fused by reciprocal rank; a KPI tile whose label the question names gets a third ranking; abbreviations are expanded from the report's own glossary; follow-ups are rewritten into standalone questions. The top 12 blocks and the matching table quantities go to the model.
3. **The model chooses and words.** It returns a JSON plan: sentences, the evidence ids behind each, and calculations over table quantities or figures it copies from a block. It never writes a page number and never computes.
4. **Code checks what can be checked deterministically.**

| Risk | Check |
|---|---|
| Wrong page | pages are inserted from node metadata; any page outside the retrieved evidence fails the answer |
| Wrong figure, unit or sign | every figure must appear in its cited block with the same magnitude and sign; numbers in the question are premises, not evidence |
| Wrong arithmetic | a Decimal calculator; incompatible units or periods stop the calculation |
| Invented calculation input | a copied figure is used only if its number, unit and year are found in the cited block; otherwise the calculation is refused and the answer says what is missing |
| Invalid or unsupported numerical claims | a sentence that fails any check gets one repair call, then is dropped; the answer becomes `partial` and says so |

**Grounded refusal** is a separate decision. The model may answer `not_in_report` or `ambiguous`; code sets the final status from that plan and from the checks, and an answer whose every sentence fails becomes a refusal rather than a guess. These checks cover numbers, units, signs, pages and calculations. A qualitative sentence (a ranking, a direction, a description) is tied to the blocks it cites but its wording is not verified by code; evaluation reads it.

The layers cover each other's blind spots: extraction makes a figure's unit and period explicit, which is what lets code verify it, and the citation lets the analyst check in one click what code cannot. Document-specific configuration is small and isolated (page map, two title font sizes, the statement form-code pattern, the glossary page marker); the rest applies to other Vietnamese reports.

## What I tried that did not work
- **Cross-encoder reranking.** Problem: fused retrieval sometimes ranked the gold page low. Choice: BGE-reranker-v2-m3 over 20 to 40 candidates. Measurement (run m4, 85 answerable dev+val items): gold page in the final blocks for 0.894 of items with the reranker, 0.882 without, at about 60 s of CPU per question. Decision: removed; p50 latency fell from 30 s (val1) to about 2 s, and the evidence count is now set by a fixed rule on recall (k = 12, run m7).
- **Chunk context from the chapter name only.** Problem: the Retail block's loan balance (`Dư nợ vay 328,1`, p.59) was refused because the tile did not say it was Retail. Choice: carry each page's title into every node. Measurement: sq-07 answered, and the same mechanism labels segment, subsidiary and statement pages. Decision: kept; it is part of every node's text.
- **Number checks on raw digits.** Problem: reading every dev answer (run dev9) showed `179,501 nghìn tỷ` accepted against evidence `179.501 tỷ`, a factor-of-1,000 slip. Choice: compare canonical numbers together with magnitude and sign, verify the unit and year of copied figures, and stop calculations with missing inputs. Measurement: each case has an adversarial test; the re-run (dev11) exposed one false rejection, a parenthetical read as a negative sign, fixed and tested. Decision: kept.

Other experiments (tokenizers, fusion parameters, chunk sizes, vector database, sparse retrieval) are in DESIGN §17.

## Evaluation
- Results (code at commit `5c5f329`, claude-haiku-5-5, 2 vCPU; runs `dev13`, `val13`, `q10k`, `test1b`, `test2` in `runs/`, rescored copies end in `r`):

| Set | Pass | Fact correct | Page hit | Unanswerable refused |
|---|---|---|---|---|
| **test2 (independent)** | **22/22** (21/22 as first scored) | 18/18 | 18/18 | 4/4 |
| q10 | 10/10 | 9/9 | 9/9 | 1/1 |
| test, after its fixes (not independent) | 22/22 (19/22 before) | 18/18 | 18/18 | 4/4 |
| val | 33/34 (32/34 as first scored) | 27/28 | 27/28 | 5/5 |
| dev | 68/70 | 55/57 | 56/57 | 12/12 |

  No answer on any set cites a page outside its evidence, and no unanswerable question was answered. I read every q10, test and test2 answer against its cited page.
- **One scorer change, made after test2 and applied to every run.** The one test2 failure was a correct answer: "hơn 1 triệu thẻ" for the gold `1.000.000`. The scorer's rule (the gold value stated in another unit or more precisely counts) did not cover round counts; it now does (`hơn 303.000` is met by `303.446`, `1.000.000` by `1 triệu`, never by another magnitude). `tcb eval --rescore` re-scores the saved answers without calling the model. It changed two items, both to pass (test2 `y-visaeco`, val `n-tinchap`), and nothing else. Both scores are shown above.
- Method: deterministic scorers, no LLM judge. A fact counts only with the magnitude the gold prints (`100 tỷ` does not satisfy `100 triệu`); citations must be pages of retrieved evidence; refusals are scored on unanswerable questions; each failure is attributed to its first failing stage from the trace. Gold facts are machine-checked against the PDF text of their page (`tcb vset check`, calculations recomputed); I reviewed every item.

| Set | Items | Role |
|---|---|---|
| dev | 70 | development and debugging |
| val | 34 | configuration choices |
| q10 | 10 | published sample; also used for debugging |
| test | 22 | written after the last code change, on pages no other set uses, run once (test1); its failures drove the fixes below, so it no longer counts as independent |
| test2 | 22 | written after those fixes, on pages no other set uses, frozen before its first run |

- What test1 found and what changed. Reading the answers found three failure modes, one of them in an answer the scorer passed. Each fix is general and tested on other pages.
  - **A change direction drawn as a graphic** (`x-ees`, scored a pass but wrong): the page draws ▲ before `7%`, the text has no direction, and the answer said "giảm 7%". The triangles are vector paths, so extraction now reads them and writes a marker after the figure (90 triangles on 30 spreads, 63 next to a figure).
  - **A substitute metric** (`x-credcost` and its follow-up): the short tile `Chi phí tín dụng | 0,6%` ranked below the 12 blocks given to the model, behind statement rows on provision expense, and the answer gave the provision expense, correctly cited. A tile whose label the question names now gets its own ranking (E8), and the prompt forbids answering one metric with another.
  - **An over-refusal** (`x-masan-fee`): the gold page ranked first, but the note heading had replaced the sub-table caption, so the rows no longer said they were service income and the model declined. The caption is now kept as the table's first row and repeated in later chunks.
  - On dev: a total given as its two components (`t-18`) led to a prompt rule to sum the components. `n-equity` cited p.256, which prints the same equity as 179.501 tỷ đồng: after review I added it as an alternative gold page and rescored.
- Two small changes after the runs, neither altering a figure or a page: `điểm phần trăm` now has the magnitude of `%` (one dev answer stating a correct percentage-point change had been dropped by the check), and the drawn-arrow marker is removed once a verb states the direction. The few dev and val misses are logged in DESIGN §17.3.
- What the numbers made you change: the reranker removal and k = 12 (E4); page-title context (sq-07); literals for arithmetic on prose figures (val failures); the magnitude, sign and canonical-number checks (reading dev9); one paragraph per answer with each figure stated once (reading q10e); direction markers, KPI label ranking and sub-table captions (test1).

## Cost and latency
- Ingestion: parse 48.2 s (including 18 s to read the drawn triangles), embedding 79.4 s on a T4, BM25 1.2 s for 1,445 nodes. One-off, no API cost.
- Per query (2 vCPU, no GPU; claude-haiku-5-5 at $0.10 / $0.50 per million input / output tokens): p50 2.2 to 2.5 s and p95 2.8 to 4.6 s across the five runs (9.4 s on test1b, two slow model calls); about $0.001 per question (8k to 12k input and 180 to 250 output tokens, 1.0 to 1.2 model calls including follow-up rewrites and repairs). Median stage times (dev13): query embedding 0.51 s, dense search 0.08 s, BM25 under 0.01 s, model 1.72 s. BGE-M3 loads once at start (22 s, 2.7 GB peak memory).
- Models: BAAI/bge-m3 for embeddings; claude-haiku-5-5 for every reported number. Run artifacts with commit, model and hardware are in `runs/`.

## What I deliberately did not build
- **Typed quantities from tables whose headers cannot be resolved** (31 multi-level note tables such as provision matrices and maturity gaps). They stay searchable as evidence text but feed no calculation: I preferred no arithmetic over arithmetic on a mis-labelled column. Next step: header recovery with a layout model.
- **An entailment check on qualitative wording.** Code verifies every figure, unit, page and calculation; the words around them are bound to cited evidence by the plan and measured in evaluation. Next step: a per-sentence entailment call on the cited block, validated against human labels.
- **OCR or a vision model.** An audit of the 428 raster images covering at least 1 % of a half-page found photographs, portraits, logos and backdrops; text over badges and diagrams is vector text and is extracted, and the KPI direction triangles are vector paths, read without OCR. Only the labels of one diagram (p.10) are carried by pixels alone and are not read. Next step: OCR for image-only text if a later report needs it.
- **An LLM judge**, which needs human labels before it can be trusted, and an agent framework, which the single-document, single-pass pipeline does not need.
- Outside the brief's scope: authentication, multi-user serving and deployment.

## With 10x time and budget
1. An analyst-labelled evaluation set with a held-out split per release, and an LLM judge calibrated on it.
2. Per-sentence entailment checks on qualitative claims.
3. Header recovery for multi-level tables so every table exposes typed quantities.
4. A Vietnamese embedding and sparse-retrieval benchmark on that set.
5. Multi-document support: one manifest and page map per report, retrieval scoped by issuer and year.

## Demo video
<!-- link -->

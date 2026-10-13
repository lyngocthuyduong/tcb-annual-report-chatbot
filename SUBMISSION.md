# Submission

An analyst pastes the answer into an email, so they must be able to check it in seconds: the figure, its unit and the page it came from. Every design choice below serves that check. Figures, units, pages and arithmetic are produced or verified by code; the language model only chooses which retrieved evidence answers the question and words the answer.

## How to run
- Clean machine: `mkdir -p out && cp .env.example .env`, put the key in `ANTHROPIC_API_KEY` (or choose another `PROVIDER`), then `docker compose up --build` → http://127.0.0.1:8000 (UI, `POST /chat`, `/docs`).
  - Prerequisites: Docker Compose v2, about 6 GB of RAM for Docker, internet for the first build. Creating `out/` first keeps it writable for the container's user.
  - Before serving, the app validates the shipped index and the PDF and exits with a named error (code 3) on any mismatch.
  - The full path (build, model check, index validation, tests, answer, refusal, q10) passed `./smoke.sh` at commit `9ee2ce1` (`runs/smoke-9ee2ce1.txt`).
- Different key or local model: set `PROVIDER=openai|anthropic|ollama` and the matching key or model in `.env`. There is no automatic fallback between providers.
  - Local model: run Ollama on the host (`ollama pull qwen2.5:7b`), set `PROVIDER=ollama`, then `docker compose up --build`; `tcb smoke` sends it one fixed question.
  - I planned a local run too, but on the machines I had (2 vCPU, no GPU) a 7B model needs minutes per answer at these prompt sizes, so the results below use claude-haiku-5-5 and the Ollama adapter is tested against a mocked server.
- Batch run: put the question file in `data/` and run `docker compose run --rm app tcb batch data/questions.json --out out/runs/x`; `answers.jsonl` holds each answer, status, citations and trace.
  - Accepts the brief's `sample_questions.json` shape, a list of strings, JSON Lines or one question per line; a follow-up keeps its context through a `history` list or a shared `session` id, and a file with no questions is an error.
  - Scoring: `tcb eval <file>`.
- Environment variables: `PROVIDER`, `OPENAI_API_KEY`, `OPENAI_MODEL`, `ANTHROPIC_API_KEY`, `ANTHROPIC_MODEL`, `OLLAMA_BASE_URL`, `OLLAMA_MODEL`, `OLLAMA_NUM_CTX`, `LLM_TIMEOUT_S`, `LLM_MAX_OUTPUT_TOKENS`, `INDEX_DIR`, `APP_HOST`, `APP_PORT`, `TRACE_DIR`, `LOG_LEVEL` (described in `.env.example`); outside Docker only: `TCB_MODELS_DIR` (where the embedding model is stored), `POPPLER_BIN` (folder of the poppler tools), `TCB_DEVICE` (`cuda` to speed up an index build). Embedding model, tokenizer and chunking are fixed in code and pinned in the index manifest.
- Shipped index: `idx/` (39 MB).

## Depth track(s) chosen
**Document intelligence** and **Retrieval and reasoning.** In this report a wrong figure or page usually starts before the model is called: two-page spreads, multi-column prose, KPI tiles split into separate text boxes, statement headers spread over several lines, Vietnamese number formats. Document intelligence makes the right evidence findable and machine-readable; retrieval and reasoning keep the answer to what that evidence proves. An evaluation harness ties every failure to the stage that caused it.

## Architecture
Four layers, each responsible for one part of the check.

1. **Extraction keeps structure.** Each spread is split into printed pages (380 of 393 numbered from their footer, 13 inferred), so `[tr. N]` is the page the analyst sees. Layout, not text order, defines 1,445 typed nodes: KPI tiles reassembled as label, figure, unit and change, with the up or down triangle drawn beside a change kept as a marker (`7% (▲)`); statement rows rebuilt from column positions with each column's period and unit (the form code printed above the notes pages once became a column header, so a 2024 related-party figure read `B05/TCTD-HN: 710.409`); prose read column by column; glossary entries. Each node carries its page title as context (`[… › Khối Ngân hàng Bán lẻ (RBG)] Dư nợ vay | 328,1 | Nghìn tỷ đồng`).
2. **Retrieval finds evidence.** BM25 over a Vietnamese tokenizer that keeps numbers whole, plus BGE-M3 dense vectors, fused by reciprocal rank; a KPI tile whose label the question names gets a third ranking, because a two-word tile such as `Chi phí tín dụng | 0,6%` otherwise loses to long statement rows on provision expense; abbreviations are expanded from the report's own glossary; follow-ups are rewritten into standalone questions. The top 12 blocks and the matching table quantities go to the model.
3. **The model chooses and words.** It returns a JSON plan: sentences, the evidence ids behind each, and calculations over table quantities or figures it copies from a block. It never writes a page number and never computes.
4. **Code checks what can be checked deterministically.**

| Risk | Check |
|---|---|
| Wrong page | pages are inserted from node metadata: the first block listed for the sentence and the blocks that print its figures; any page outside the retrieved evidence fails the answer |
| Wrong figure, unit or sign | every figure must be printed in a cited block with the same magnitude, and with the same sign unless a word carries it (`lỗ`, `chi phí`); an amount restated as `tương đương` must be the same amount; numbers in the question are premises, not evidence. The check ties a figure to its block, not to a row or year inside it: the citation shows the analyst which |
| Wrong arithmetic | a Decimal calculator; incompatible units or periods stop the calculation |
| Invented calculation input | a copied figure is used only if the cited block prints it with the same sign, currency and magnitude, under the declared year (the year printed before it in the same sentence or cell, or `năm trước`); otherwise the calculation is refused and the answer says what is missing |
| Invalid or unsupported numerical claims | a sentence that fails any check gets one repair call, then is dropped; the answer becomes `partial` and says so |

**Grounded refusal** is a separate decision. The model may answer `not_in_report` or `ambiguous`; code sets the final status from that plan and from the checks, and an answer whose every sentence fails shows no figure and says it could not verify one, rather than that the report lacks it. These checks cover numbers, units, signs, pages and calculations. A qualitative sentence (a ranking, a direction, a description) is tied to the blocks it cites but its wording is not verified by code; evaluation reads it.

The layers cover each other's blind spots: extraction makes a figure's unit and period explicit, which is what lets code verify it, and the citation lets the analyst check in one click what code cannot. Document-specific configuration is small and isolated (page map, two title font sizes, the statement form-code pattern, the glossary page marker); the rest applies to other Vietnamese reports.

## What I tried that did not work
- **Cross-encoder reranking.** Problem: fused retrieval sometimes ranked the gold page low. Choice: BGE-reranker-v2-m3 over 20 to 40 candidates. Measurement (run m4, 85 answerable dev+val items, on an earlier index): gold page in the final blocks for 0.894 of items with the reranker over 30 candidates and 0.882 for the fused top 10 without it, while reranking 30 candidates took about 60 s of CPU per question. Decision: removed, and the evidence count set by a fixed rule on recall (k = 12, run m7). Answer latency was p50 30 s with the reranker over 10 candidates (val1) and is about 2 s in the final runs, which also differ in index and evidence count.
- **Chunk context from the chapter name only.** Problem: the Retail block's loan balance (`Dư nợ vay 328,1`, p.59) was refused because the tile did not say it was Retail. Choice: carry each page's title into every node. Measurement: sq-07 answered, and the same mechanism labels segment, subsidiary and statement pages. Decision: kept; it is part of every node's text.
- **Number checks on raw digits.** Problem: reading every dev answer (run dev9) showed `179,501 nghìn tỷ` accepted against evidence `179.501 tỷ`, a factor-of-1,000 slip. Choice: compare canonical numbers together with magnitude and sign, verify the unit and year of copied figures, and stop calculations with missing inputs. Measurement: each case has an adversarial test; the re-run (dev11) exposed one false rejection, a parenthetical read as a negative sign, fixed and tested. Decision: kept.

Other experiments (tokenizers, fusion parameters, chunk sizes, vector database, sparse retrieval) are in DESIGN §17.

## Evaluation
- Method: deterministic scorers, no LLM judge.
  - Facts: a fact counts only with the magnitude the gold prints (`100 tỷ` does not satisfy `100 triệu`); the gold stated more precisely or in another unit counts.
  - Pages: citations must be pages of retrieved evidence. Page hit counts the declared gold pages and any page that prints all the gold figures (`widen_gold`), since the analyst can check the figure there; hits on declared pages alone are reported too.
  - Refusals are scored on unanswerable questions, and each failure is attributed to its first failing stage from the trace.
  - Gold facts are machine-checked against the PDF text of their page (`tcb vset check`, calculations recomputed), and I reviewed every item and read every q10, test and test2 answer against its cited page; the notes saved inside `runs/test1` predate that review.
  - Each run is one sample at the provider's default sampling; q10 and test2 were run several times to show the spread.

| Set | Items | Role |
|---|---|---|
| dev | 70 | development and debugging |
| val | 34 | configuration choices |
| q10 | 10 | published sample; also used for debugging |
| test | 22 | written after the last code change, on pages no other set uses, run once (test1); its failures drove the fixes below, so it no longer counts as independent |
| test2 | 22 | written after those fixes, on pages no other set uses, frozen before its first run |

- Results on the 10 published questions: 10/10 in each of three runs (9/9 facts, 9/9 pages, the unanswerable one refused). All sets at commit `86d5252`, claude-haiku-5-5, 2 vCPU (`runs/dev15`, `val15`, `q10o` to `q10q`, `test2d`, `test2e`; earlier runs in `runs/history/`):

| Set | Pass | Fact correct | Page hit (declared pages only) | Unanswerable refused |
|---|---|---|---|---|
| **test2 (independent)** | **22/22** on its first run (`6d84763`, `runs/test2r`); 21/22 in two later runs at `86d5252`, the one miss a calculator bug fixed since (below) | 18/18; later 17/18 | 18/18 (18/18); later 17/18 | 4/4 |
| q10 | 10/10 in each of three runs | 9/9 | 9/9 (8/9) | 1/1 |
| val | 31/34 | 26/28 | 26/28 (26/28) | 5/5 |
| dev | 68/70 | 55/57 | 57/57 (55/57) | 12/12 |

  No answer on any set cites a page outside its evidence, and no unanswerable question was answered. The test2 miss in both runs (`y-followers`, a sum of two follower counts) and one val miss (`n-hr-diff`, a difference of headcounts) came from one cause: the unit-rescaling step added to `sum` and `diff` refused count units such as `lượt theo dõi`. It is fixed after these runs; replaying the saved plans through the fixed calculator gives 264.000 and 857, and changes no other saved calculation. Every dev and val miss is listed with its question, answer and cause in DESIGN §17.3.
- **One scorer change, made after test2 and applied to every run.** The one test2 failure was a correct answer: "hơn 1 triệu thẻ" for the gold `1.000.000`. The scorer now reads a word scale as an exact conversion (`1 triệu` = `1.000.000`, never `600.000`) and a gold count ending in three zeros as rounded to the thousand (`hơn 303.000` is met by `303.446`). `tcb eval --rescore` re-scores the saved answers without calling the model; it changed two items, both to pass (test2 `y-visaeco`, val `n-tinchap`), and nothing else.
- What the numbers made you change:
  - E4 set the evidence count (k = 12) and the reranker was removed (above).
  - Reading every answer, not only the scores, found the costly errors. On test1 an answer the scorer passed said "giảm 7%" where the page draws an up triangle: the direction exists only as a vector path, so extraction now reads those paths into markers (63 in the index). The same run gave provision expense for a question on credit cost (hence the KPI label ranking) and refused one note-table question because the note heading had replaced the sub-table caption (now kept as the table's first row).
  - On the runs at `6d84763`, an answer put `834.241.663 triệu đồng` next to `tương đương 776,5 nghìn tỷ đồng`: two different figures, both printed in the report, so every number check passed. Restated amounts are now compared after calculation.
  - Gold changes after review are few and named in the item notes: two dev items gained an alternative page that prints the same figure (`n-equity`, `h-kpi-pbt`), and the scorer change above.

## Cost and latency
- Ingestion: parse 48.2 s, embedding 79.4 s on a Colab T4 GPU, BM25 1.2 s for 1,445 nodes (`idx/report.json`). One-off, no API cost.
- Per query (2 vCPU, no GPU; claude-haiku-5-5 at $0.10 / $0.50 per million input / output tokens): p50 1.9 to 2.3 s and p95 2.4 to 4.2 s across the final runs (6.4 s on one q10 run, one slow model call); about $0.001 per question (about 11.6k input and 230 output tokens on dev, 1.0 to 1.15 model calls including follow-up rewrites and repairs). Median stage times (dev15): query embedding 0.51 s, dense search 0.09 s, BM25 under 0.01 s, model 1.54 s. BGE-M3 loads once at start.
- Models used: BAAI/bge-m3 for embeddings; claude-haiku-5-5 for every reported number. Run artifacts with commit, model and hardware are in `runs/`; earlier runs cited here are in `runs/history/`.

## What I deliberately did not build
- **Typed quantities from tables whose headers cannot be resolved** (31 multi-level note tables such as provision matrices and maturity gaps). They stay searchable, and a figure in them can still be copied into a calculation as a checked literal, but they expose no automatically parsed quantities: I preferred that to arithmetic on a mis-labelled column. Next step: header recovery with a layout model.
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

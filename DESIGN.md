# DESIGN — Vietnamese chatbot over the Techcombank Annual Report 2025


---

## 1. Goals, constraints, non-goals

**Goal.** Answer Vietnamese questions about one document (Techcombank Annual Report 2025, 196 two-page spreads) with answers that are (a) grounded in the report, (b) cited by printed page `[tr. N]`, (c) numerically correct when arithmetic is needed, and (d) refused when the report does not contain the answer.

**Fixed by the assessment / by me and not to be changed without asking**

| Item | Decision |
|---|---|
| Embedding model | `BAAI/bge-m3`, revision pinned in `conf.EMBED_REV` |
| Reranker | none (measured and removed, §17.1) |
| Orchestration | LlamaIndex for ingestion/vector retrieval only. No agent framework |
| LLM modes | Online default (OpenAI or Anthropic), explicit Offline profile (Ollama `qwen2.5:7b`). **No automatic fallback between providers or modes** |
| Depth tracks | Document Intelligence + Retrieval + Evaluation |
| Index | Built once, committed as `idx/`, validated fail-closed at start-up. Never rebuilt or downloaded at run time |
| Time | 4-day sprint; minimal code, no speculative abstractions |

**Non-goals.** Multi-document support, an LLM judge, OCR of image-only content (inventoried but not read), streaming answers, authentication, multi-process serving, runtime fine-tuning.

**Honesty rules baked into the project.** A mock pass is never reported as proof that a real model works. The hash embedder exists only in `tests/` and writes to a tmp dir; the production validator rejects any index not built with the pinned BGE-M3 revision. Eval items count only when a human set `verified: true`. Items I drafted are `verified: false`.

---

## 2. The document and why it is hard

Facts measured on the real PDF:

- 197 PDF pages = cover + 196 **spreads** (two printed pages side by side). The printed page number p lives on pdf page `(p+2)//2` if p is odd (right half) else `p//2+1` (left half). A citation unit is therefore a **half-page**: 393 halves.
- Printed page numbers are read from footers on 380 halves; 13 divider/blank halves are inferred from neighbours (node-level: 1,439 nodes footer-sourced, 6 inferred).
- Content kinds: narrative prose, **KPI tiles** (big number + caption), **financial-statement tables** (forms B01–B05 and notes), and a **glossary** (163 terms).
- Vietnamese numbers: `.` is the thousands separator and `,` the decimal (`1.192.344.137`, `14,6%`, negative as `(181.150)`). Units are in headers (`Triệu đồng`, `Nghìn tỷ đồng`). The same quantity appears at different scales: total assets is `1.192 nghìn tỷ đồng` on p.5 and `1.192.344.137 triệu đồng` on p.297.
- Images: 428 embedded images on 117 pages, counted in `report.json`. A one-off visual audit (13 Oct 2026; image boxes from the PDF, every image covering ≥ 1 % of a half-page) found no chart, table or figure that exists only as an image: 82 are full-page backdrops (73 with the page's real text drawn on top), 159 are logos and icons under 1 % of the page, and the 126 mid-size images with no text over them are portraits, photos, chapter numbers and adverts. Award badges and the six-pillar diagram (p.46) are vector text over an image and are read. One diagram holds labels only as pixels (p.10, the 'Customer Brain' labels). No OCR or vision model is used.

Failure modes that a naive "split text, embed, ask GPT" pipeline hits on this document, and where this design answers each:

| Failure | Where handled |
|---|---|
| Wrong page cited (spread vs printed page) | §4.2 page map; citation page comes from node metadata only (§8) |
| Table values scrambled / columns lost | §4.3 coordinate-based table parser + `quality` flag |
| `1.192` tokenised as `192`, `14,6%` as `14` and `6` | §4.5 tokenizer `tcb-vi-v1` |
| LLM does arithmetic wrong | §9 calculator; LLM only picks operation and input ids |
| LLM invents a number or page | §8 deterministic validation |
| Confident answer to an unanswerable question | §10 refusal policy, fail closed |
| Follow-up ("Còn năm 2024?") loses context | §11 rewrite |
| Can't tell which stage lost the answer | §14 stage attribution |

---

## 3. System overview

BUILD ONCE (networked machine, real BGE-M3)
```
data/ar.pdf
 -> pdftotext -bbox-layout
 -> pdf.py      spread split, printed-page map
 -> parse.py    KPI tiles, FS tables, glossary
 -> nodes.py    typed nodes + provenance
 -> LlamaIndex vector index (BGE-M3, 1024-d)
    + bm25s index (tokenizer tcb-vi-v1)
 -> idx/ + manifest.json (sha256 of every file)
```

RUN (docker compose up); the index is validated first, exit 3 on any mismatch
```
question
 -> rewrite (only for follow-ups)
 -> glossary expansion
 -> BM25 top-40  +  dense top-40
 -> RRF (k=60) -> top 12 = evidence blocks E1..E12
    + quantities Q1..Qn (high-quality tables; the 60 whose labels best match the question)
 -> LLM returns a JSON plan
    (claims -> evidence ids, calculations -> quantity ids or literals L1..Ln
     copied from evidence text and checked verbatim)
    at most 1 repair retry, same provider
 -> cite.check   ids exist, no page numbers in text,
                 every number is in cited evidence or a calc result
 -> calc.run     Decimal, unit and period checks
 -> refuse.decide
 -> render "[tr. N]" from node metadata
```

Module map (`src/tcb/`):
- ingestion: `pdf parse nodes vi`
- index: `build index`
- retrieval: `find`
- quantities: `num calc`
- LLM: `plan gen llm/`
- validation: `cite refuse`
- pipeline: `chat ask boot`
- surfaces: `api ui.html cli smoke`
- evaluation: `ev exp vset`
- environment: `conf doctor models`

The one design idea running through everything: **the LLM is not trusted with facts, pages, or arithmetic.** It only chooses *which retrieved evidence supports which sentence* and *which calculator operation on which extracted quantity*. Everything a reader can check (numbers, pages, quotes) is produced or verified by deterministic code.

---

## 4. Ingestion (document intelligence)

### 4.1 Extraction
`pdftotext -bbox-layout` (poppler) returns words with boxes. Its output is not well-formed XML, so `pdf.read_words` parses it with a regex. Poppler is located via `PATH` or `POPPLER_BIN` (portable across Windows/macOS/Linux); a missing tool produces an actionable message.

*Alternative rejected:* pdfplumber layout extraction (initial plan). On this spread layout the poppler boxes were reliable and faster, and no extra dependency was needed.

### 4.2 Spread split and the printed-page map
Each pdf page is split into a left and right `Half` at the gutter; words straddling the gutter are counted (`straddle_words`) and assigned by centre. Running furniture (headers/footers) is stripped from the body. `footer_number` reads the printed number; `build_page_map` cross-checks footers against the linear formula in §2, records `conflicts`, and **raises `PageMapError`** rather than guessing if the map is inconsistent. Divider halves without a footer get an inferred number and `page_number_source="inferred"`.

Invariant tested on the real PDF: for every node with printed page p, `pdf_page_index` and `side` equal the formula; only the cover may lack a printed page; every published gold fact (9 answerable q10 items) appears as text on its gold printed page.

### 4.3 Typed nodes
`nodes.build_nodes` yields **1,445 nodes**: narrative 983 · KPI 180 · table 119 · glossary 163. Prose is read column by column (blocks that are full-width, or that bridge two columns like a centred quote box, split a half into bands; inside a band, overlapping x-ranges form a column), and a text that starts in lower case is appended to the nearest earlier text that ends mid-sentence, so a sentence wrapping into the next column or around a KPI tile stays whole. KPI tiles are ordered label → figure → unit → change; tiles with fewer than two words are dropped when they hold three or more numbers (chart ticks, contents pages) or merged into the page text otherwise. Table headers are reduced to period + unit when they contain one; a table is `high` quality only if every column has a period. Every node text starts with `[chapter › page title › subtitle]`: titles are read from font height (≥ 35 pt title, 27 to 35 pt subtitle, top 30 % of the half), a title carries over to later pages until the next one, a left-half subtitle (a subsidiary or segment name) replaces the carried one, recurring labels (`Điểm nhấn 2025`, `Trọng tâm năm 2026`) stay local, and statement tables carry their own title (an untitled table on the next page inherits it). A node never spans two printed pages. Each carries `chunk_id` (stable hash), `printed_page_number`, `pdf_page_index`, `side`, `page_number_source`, `section`, `content_type`, `bbox`, and for tables `table_id`, `form_code`, `unit_header`, `rows`, `quality`.

- **Narrative:** sentence-aware packing to ≤ 220 syllable-words (`CHUNK_MAX_WORDS`; median ≈ 175, max 226 because a single sentence is never cut mid-way unless it alone exceeds the budget). Chunk size is an experiment (E6) but any non-default index is rejected by `serve`/`validate`.
- **KPI tiles:** grouped by proximity into one node per tile (label + value + caption) so "Tỷ lệ CASA 40,4%" stays one retrievable unit.
- **Glossary:** one node per term; also exported to `gloss.json` for query expansion (§6.2).
- **Tables:** see below.

### 4.4 Financial-statement tables
Tables are rebuilt **from coordinates**, not from text order: numeric words are anchored into columns by right edge (clustered within 14 pt), the first column is the label with indentation, a narrow column is the *Thuyết minh* note number, and wrapped labels are re-joined. Nil cells (`-`) become `None`; parentheses mean negative. Rows are chunked 12 per node and rendered as `label [TM n] | <header>: value; ...` so the text stays searchable and unit-bearing.

Header handling (fixed this session after the income statement and cash flow were found to be mis-parsed) [verified by regression tests on the real PDF]:
- A first row whose cells are all years is promoted to headers `Năm 2025` / `Năm 2024`, and the column unit is appended (`Năm 2025 Triệu đồng`).
- A unit-only row (`Triệu đồng Triệu đồng`) fills the unit list instead of leaking into the first label.
- A repeated column-header row inside a table (dates, units, *Thuyết minh*, years) is dropped; a CAPS section title glued to it is dropped with it; a CAPS section title above a normal row becomes its own row with empty cells (e.g. `LƯU CHUYỂN TIỀN TỪ HOẠT ĐỘNG KINH DOANH`).
- A pending label starting in lower case is the wrapped tail of the previous row's label.

Second pass (13 Oct 2026, after reading the notes tables row by row):
- The form code printed at the top right of every notes page (`B05/TCTD-HN`) sat above the last number column and became its header: on p.369 the 2024 column read `B05/TCTD-HN: 710.409`. The upward search for header rows now stops at the line that carries the form code, and the code is never a header word.
- A row whose cells are all years (`Bên liên quan | 2025 | 2024`) is the header of the sub-table below it, wherever it appears; one page can stack several small tables whose year columns sit at different x positions, so each row's cells are re-keyed by year. p.369 and p.370 (related parties, remuneration) became `high`.
- The unit line under a year row (`Triệu đồng Triệu đồng`) sets the column unit; a sentence above the table (`thuế suất 20% (2024: 20%)`, p.351) had made the columns `%`. The unit stored for a column is the one its cleaned header shows; `hợp đồng` is not the unit `đồng`.
- Note tables are named by the nearest numbered heading above them (`6. TIỀN GỬI TẠI NGÂN HÀNG NHÀ NƯỚC VIỆT NAM`), carried to the next pages until the next heading. Before, the second table on a page took the last rows of the table above as its title.
- The statement band repeated at the top of 80 pages (`THUYẾT MINH BÁO CÁO TÀI CHÍNH HỢP NHẤT (tiếp theo)`, the period line, the form code) is page furniture: it is removed from narrative and table text, and the statement name goes into the context prefix (`[Báo cáo tài chính › Thuyết minh báo cáo tài chính hợp nhất]`). A narrative chunk that only repeats a table's own title is not indexed.

**Quality flag.** `quality="high"` only if every column header carries a date or `Năm YYYY`, or the table has ≤ 3 clean columns. Currently 66 high / 53 low chunks (36 of 67 tables). The primary statements (printed pages 297–302, forms B02 to B04: financial position, off-balance items, income, cash flow) are all high. Low tables (multi-column notes: provisions, segment, related-party, maturity gaps) keep their text view and are **excluded from calculator quantities** rather than guessed.

*Alternative rejected:* marking every table high and letting the LLM sort out columns — that is exactly how wrong numbers get produced from scrambled headers.

### 4.5 Tokenizer `tcb-vi-v1` (BM25)
`bm25s`' default tokenizer splits `1.192` into `192`. `vi.tokens` instead keeps dates, form codes (`B03/TCTD-HN`) and numbers whole, emits the canonical form too (`1.192.344.137` → `1192344137`, `14,6%` → `14.6%`), and emits a diacritic-folded copy of every word, so `Tăng trưởng` and `tang truong` both match. Exact BM25 behaviour on numbers is why hybrid retrieval beats dense-only on a report full of figures (hypothesis; measured by E2/E3, **[NOT RUN]**).

---

## 5. The index and its integrity

`build.build` writes `idx/`: `llamaindex/` (vector store), `bm25/` (+ `ids.json`), `nodes.jsonl`, `tables.jsonl`, `pages.json`, `gloss.json`, `report.json`, and `manifest.json`.

The manifest records: index format version, embedding model + revision + dimension, tokenizer name, chunking (`narrative_max_words`, `table_rows_per_chunk`), BM25 k1/b, package versions, PDF sha256, counts, build time/hardware, and a **sha256 for every other file**.

`index.validate` runs before any model loads and fails closed with a code and a remediation (process exit **3**) when: manifest missing/corrupt; format mismatch; embedding model or revision differs from `conf`; chunking differs (experimental index); tokenizer differs; unlisted or missing files; any hash mismatch; node counts disagree between `nodes.jsonl`, BM25 ids and the LlamaIndex docstore. `--snapshot` additionally checks the baked model snapshot's `.rev`.

Why: a stale or half-built index silently degrades retrieval and nobody notices. Refusing to start is cheaper than shipping wrong answers. `.gitattributes` marks `idx/**` as `-text -diff` so Git's CRLF conversion on Windows cannot change a hash.

The validator and build are tested with a hash embedder in tmp, and the validator is tested to *reject* that index.

---

## 6. Retrieval (`find.py`)

### 6.1 Pipeline
1. **BM25** over `tcb-vi-v1` tokens → top 40 (`TOP_BM25`).
2. **Dense** (BGE-M3 via LlamaIndex) → top 40 (`TOP_DENSE`).
3. **RRF** with k = 60 (`RRF_K`) over the two rankings; the top **12** (`FINAL_EVIDENCE`, E4 run m7) go to the LLM, further limited by a 22,000-character prompt budget (`PROMPT_EVIDENCE_CHARS`) so a local 8k-context model is never silently truncated.

There is no cross-encoder and no score threshold (§17.1). `Result` keeps every stage's ranking (`bm25`, `dense`, `fused`, `ranked`) and the seconds spent on BM25, query embedding and dense search, so evaluation can say *which stage lost the gold page* and *where the time went* (§14).

### 6.2 Glossary expansion
If the query contains an upper-case glossary abbreviation (e.g. CASA), the report's own definition (first 100 characters) is appended in parentheses: `CASA (Tiền gửi không kỳ hạn…)`. The expanded query is used by both stages. E5 (run m4): no item lost, none gained; kept because abbreviations such as RBG and CASA need it on unseen questions.

### 6.3 No weak-retrieval retry
An earlier design retried a query once when the top cross-encoder score fell below τ_low (E7). E7 proposed τ = 0.599, but the retry meant a second full search on exactly the weak and unanswerable questions, and no experiment showed it recovering a page. It was removed together with the reranker; refusal relies on the plan and the validators (§10).

### 6.4 Why this shape
- BM25 + dense because figures/abbreviations need lexical exactness while paraphrased narrative needs semantics.
- RRF because it needs no score calibration between BM25 and cosine; k = 60 is the literature default and is only changed if E3b shows ≥ 5 points of R@20 on **every** set.
- Top 10 fused blocks, no reranker (E4, run m4: dev+val, 85 answerable items): a cross-encoder over 20 to 40 candidates found a gold page for 0.894 of items, the fused top 10 for 0.882, i.e. one item, at about 2 s of CPU per candidate pair on 2 cores (≈ 60 s per question at 30). With 10 candidates and 10 blocks the reranker would only reorder them.
- Tokenizer stays `vi`: E2 favoured `default` by one item at R@20 (0.894 vs 0.882) while `vi` is ahead at R@5/R@8, so no rebuild.
- One retrieval path: no HyDE, multi-query or agents; each adds LLM calls and a failure surface.

---

## 7. Generation: the plan

### 7.1 Plan schema (`plan.py`)
The LLM returns **one JSON object**:

```
status: answered | partial | not_in_report | ambiguous
sub_questions: [str]
claims: [{text_vi, evidence:[E#], calc_refs:[C#]}]
calculations: [{cid, op, inputs:[Q#], to?}]     op ∈ growth_pct diff ratio share_pct cagr sum convert
clarification_vi: str?
```
Evidence blocks are `[E1]…[E8]`; extractable numbers from high-quality tables are offered as `[Q1]…[Qn]` (`label | period | value unit`, capped at 60). Claim text may contain `{C1}` placeholders where a calculator result is inserted. The system prompt (Vietnamese) forbids outside knowledge, requires numbers copied verbatim in Vietnamese format, forbids writing page numbers, forbids doing arithmetic, and tells the model to choose `not_in_report` rather than guess.

### 7.2 Providers (`llm/`)
| Provider | Structured output | Notes |
|---|---|---|
| OpenAI | `response_format: json_schema` | SDK retries off (`max_retries=0`) so retry policy is ours |
| Anthropic | forced `tool_use` with the schema | |
| Ollama | `format=<schema>`, `num_ctx` 12288 (`OLLAMA_NUM_CTX`) | default `qwen2.5:7b`; truncation guarded by the evidence budget |

Errors are typed: `ConfigError` (exit 2), `ProviderTimeout`, `ProviderUnavailable`, `ProviderRejected`, `GenerationError`; every message names provider and model. An `error` answer is returned to the user as such. **No other provider is ever tried.** Token usage, latency and a list-price cost estimate (`conf.PRICES`) are recorded per call.

### 7.3 One repair retry (`gen.py`)
If the reply is not parseable JSON for the schema, or `cite.check` finds problems, the same provider is called **once more** with the previous output and a short list of the specific errors. Whatever is still invalid after that is dropped by the pipeline (§8), not retried. Bound: ≤ 2 generation calls + ≤ 1 rewrite call per question.

---

## 8. Deterministic validation and citation (`cite.py`)

`cite.check(plan, evidence, quantities, question)` per claim:
1. Every `evidence` id exists among the retrieved blocks and has a printed page.
2. Every `{C#}`/`calc_refs` id exists among the plan's calculations; calculation inputs exist among the extracted quantities.
3. The text contains **no** page reference (`tr. 5`, `trang 5` …) — pages are never LLM output.
4. **Every number in the claim text must appear in the cited evidence** (or in the raw cells of quantities used by the claim's calculations, or in the question), compared in canonical form so `14,6%` ≡ `14.6`. Integers ≤ 12 are exempt (list counts, months), a documented gap.
5. **Magnitude.** A number printed with a magnitude in the cited evidence (`1.192 | Nghìn tỷ đồng`, `18 triệu`, `40,4%`, or the unit of a table column, `31/12/2025 Triệu đồng: 1.192.344.137`) must carry the same magnitude in the claim. `1.192 tỷ đồng` or a bare `1.192` fails although every digit matches: that is the silent factor-of-1,000 error.
6. At least one evidence id or calculation per claim.

After the checks, **one statement per figure**: a claim whose figures were all stated by an earlier claim (the same number from another page, or the same figure at another rounding or unit: `1.192 nghìn tỷ đồng` = `1.192.344 tỷ đồng` within half a unit of the rounded one) is left out, recorded in the trace as `duplicates_removed`. A claim with a new figure, a calculation or no figure is kept. Run `q10e` showed the need: 4 of 10 answers restated a figure from a second page.

A claim failing any check is **dropped**, not patched. Rendering then builds `text + [tr. N]` where N is `printed_page_number` of the cited nodes, merged and sorted; calculator results use the pages of their input quantities. A final `verify_rendered` pass asserts every page in the rendered answer belongs to retrieved evidence (defence in depth; failure → `error` status).

The answer payload also lists `citations` (eid, page, pdf_page, section, chunk_id, snippet) for the UI to open the source.

---

## 9. Calculator (`num.py`, `calc.py`)

`num.from_node` converts cells of **high-quality tables only** into `Quantity(value: Decimal, dim, scale, period, page, raw)`; units/periods come from the column header (e.g. `Năm 2025 Triệu đồng` → VND, 10⁶, year 2025). Seven typed operations, all `Decimal`, all strict:

| op | Rule |
|---|---|
| `growth_pct(curr, prev)` | same dim & period kind, curr later than prev, prev ≠ 0 → `%` |
| `diff(a, b)` | same dim & kind; `% − %` → percentage points |
| `ratio(n, d)` | same dim, d ≠ 0 → `lần` |
| `share_pct(part, total)` | same dim and same period, total ≠ 0 |
| `cagr(start, end)` | years ≥ 1, start > 0, end ≥ 0 |
| `sum(qs)` | same dim and period |
| `convert(q, to)` | VND/USD rescale only (`triệu đồng` → `nghìn tỷ đồng`) |

Mismatches raise typed errors (`UnitMismatch`, `PeriodMismatch`, `ZeroDenominator`, `InvalidInput`), which the pipeline turns into "Không thể tính: …" notes and a `partial` status — never a guessed number. Display is Vietnamese (`21,8%`, `1.192.344.137`). Golden test on the real PDF: total assets growth from p.297 cells = `21,8%`, equal to the report's own p.5 KPI; income-statement growth `13,6%` (total operating income) and `18,2%` (profit before tax).

*Alternative rejected:* letting the LLM compute and then checking it — a check that recomputes is just the calculator with extra steps.

---

## 10. Refusal policy and statuses (`refuse.py`)

| Status | Meaning |
|---|---|
| `answered` | all claims kept, no calc errors, no validation leftovers |
| `partial` | some claims/calcs dropped or LLM said partial; answer plus a note on what could not be verified |
| `not_in_report` | LLM declined, or **no** claim survived validation (fail closed) |
| `ambiguous` | the question lacks entity/period; returns a clarifying question |
| `error` | provider/index/retrieval failure; message names provider/model |

Fail-closed means an answer exists only through claims that survived §8. Retrieval-score thresholds never suppress a grounded answer on their own.

---

## 11. Multi-turn (`chat.py`)

A question is treated as a follow-up only if there is history **and** it is short (≤ 9 words) or contains an anaphoric marker. One LLM call (JSON `standalone`) rewrites it using the last two turns. The rewrite is accepted only if every number in the user's question is still present in it (faithfulness guard); otherwise the deterministic fallback `"<previous standalone> — <question>"` is used. The mode (`none | llm | concat`) is recorded in the trace. A provider error during rewrite is **not** swallowed: it surfaces as an `error` answer (no silent fallback). History is *context for understanding the question, never evidence*: an earlier answer cannot be cited. Sessions live in memory (cap 200) keyed by `session_id`.

---

## 12. Surfaces

- **HTTP (`api.py`, FastAPI):** `GET /health`, `POST /chat {question, session_id?}`, `GET /evidence/{chunk_id}`, `GET /pdf`, `GET /` (single-file UI `ui.html`), `/docs`. Single process, one answer at a time under a lock. Status codes map error kinds (config/timeout/unavailable/rejected). Every answer's full trace is appended to `out/trace/chat.jsonl`; tracing failures never break an answer.
- **CLI (`tcb`):** `doctor` (diagnose this machine), `models download|convert`, `check` (gate: load BGE-M3, related text closer than unrelated, one query embedding timed, log RSS), `build`, `validate`, `ask`, `chat`, `batch`, `eval` (`--compare`, `--include-unverified`), `exp` (retrieval experiments), `vset check` (gold facts on gold pages), `smoke` (Ollama), `serve`. Exit codes: 2 config, 3 index, 4 error answer, 5 model missing, 6 smoke.
- **Boot order (`boot.py`):** read config → validate index → load models → serve. A bad index exits 3 before models are touched; a missing key exits 2 before the port opens.

---

## 13. Packaging and portability

- **Docker (`Dockerfile`, `compose.yaml`):** `docker compose up --build` is the online default. The image installs CPU torch from the PyTorch index first, then `pip install --require-hashes --no-deps -r requirements.lock` (hash-pinned, Python 3.12 linux-x86_64, no torch/CUDA); BGE-M3 is downloaded at build time at pinned revisions into `/models` with a `.rev` marker. `idx/` and `data/` are mounted, not baked, so the index can be replaced without rebuilding the image. Layers are kept lean on purpose: `COPY --chown` instead of `RUN chown -R` and no in-image weight conversion, because each would duplicate 2.3 GB of weights in a new layer (a first Docker build on a 32 GB Codespace ran out of disk exporting layers). The one-off index build runs on Colab (`TCB_DEVICE=cuda` only speeds up embedding; model and revision are the pinned ones), not inside Docker.
- **Offline mode:** `docker compose --profile offline up --build app-offline` starts Ollama, pulls `qwen2.5:7b` once, and runs the app with `PROVIDER=ollama`. Separate services, no cross-fallback.
- **Local runs (no Docker):** `pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu && pip install -e ".[dev]"`, `python -m tcb.models download`; `.env` loader (does not override real env vars); `TCB_MODELS_DIR`, `POPPLER_BIN`, `PYTHONUTF8`; `import resource` guarded for Windows; explicit UTF-8 everywhere. `tcb doctor` prints a one-screen diagnosis (python, packages, poppler, PDF hash, models, index, keys, RAM/disk, Docker) with a fix per failure.
- **Locks:** `lock.sh` produces `requirements.lock` and fails if torch/CUDA leak in.

---

## 14. Evaluation design

### 14.1 Datasets
| File | Content | Rule |
|---|---|---|
| `data/q10.json` | the 10 published sample questions (9 answerable, 1 not) | never used for tuning; additively annotated with `gold_facts`, `alt_pages` (e.g. 297 for total assets), `type`, `verified` |
| `data/dev.json`, `data/val.json` | 70 + 34 items written by hand (statement tables, KPI tiles, narrative, segments, subsidiary, governance, investor relations, glossary, calculations, follow-ups, 19 unanswerable or ambiguous) | gold checked by Claude against the PDF text of the cited page, calculations recomputed; human spot-check pending |

Unanswerable topics were chosen only if a full-text search of the report found no mention (Vietcombank, Campuchia, Bitcoin, Momo, Nasdaq, World Cup, 2027, MB Bank, Moody's; enforced by a test). Topics that **are** in the report (dividends, 2029/2030, remuneration, 2026 plan) are not used as "unanswerable". `tcb vset check` asserts every answerable item's gold facts are present, as text, on its gold pages (for calc items: the *inputs* are). The first table items were drafted by a generator from the statements; it and its CSV review loop were removed once the sets were checked, so they cannot be regenerated over by accident.

Item types: `single`, `multi`, `calc`, `followup`, `refusal`; `expect_status: ambiguous` marks questions whose right behaviour is a clarifying question; these are excluded from refusal precision/recall.

### 14.2 Scorers (deterministic, no LLM judge)
- **Fact present:** every `gold_facts` entry must appear in the answer; `a | b` = either alternative; numbers compared in canonical form.
- **Citation valid:** every cited page belongs to the evidence retrieved for that question.
- **Page hit / precision:** cited pages vs `gold_printed_pages ∪ alt_pages`.
- **Refusal:** precision and recall over unanswerable items, plus false-answer rate.
- **Cost/latency:** per question from traces (p50/p95, tokens, estimated USD).

### 14.3 Stage attribution
For each failed item, the first failing stage in this order: `system` (provider/index error) → `retrieval:recall_miss` (gold page in neither BM25 nor dense top-40) → `retrieval:fusion_drop` → `retrieval:rank_drop` (gold page ranked below the `FINAL_EVIDENCE` blocks given to the LLM) → `citation` (invalid/wrong page) → `calc` → `generation:wrong_or_missing_fact` → `refusal:over_refusal` / `under_refusal`. This is what makes "what did the numbers make you change" answerable.

### 14.4 Run artifacts
`tcb eval` writes `traces.jsonl` (full per-question trace), `results.jsonl`, `summary.json` and `config.json` (set sha256, provider, model, index identity, evidence count, RRF k, device, and **`submission_grade`**: true only with the real BGE-M3 index). `tcb eval --compare A B` reports fixed / regressed / verdict-changed items and metric deltas, and warns if the two runs differ in set, provider, model, τ, k or index.

### 14.5 Experiments (`tcb exp`, retrieval only, zero token cost)
| ID | Question | Decision rule |
|---|---|---|
| E2 | tokenizer `tcb-vi-v1` vs default vs bigram | switch only with ≥ 2 pts R@20 (a switch needs an index rebuild) |
| E3 / E3b | BM25 vs dense vs RRF vs RSF; RRF k sweep | keep k = 60 unless ≥ 5 pts R@20 on **every** set |
| E4 | how many fused blocks go to the LLM (k = 5 … 20) | smallest k within 2 pts of the best recall (run m4 measured the former reranker grid N × k instead; §17.1) |
| E5 | glossary expansion | keep only if no regression |
| E6 | narrative chunk size | needs index rebuild; experimental indexes are rejected by `serve` |

The runner *suggests* a decision by these rules and never edits `conf.py`. Order of work: experiments on dev+val → freeze → eval on val → q10 once, human-scored. Val was also used to find the three failures fixed in §17.1, so it is no longer a clean held-out set; q10 was never used for a decision.

---

## 15. Test strategy (categories are kept honestly separate)

| Marker | Needs | Proves | Does not prove |
|---|---|---|---|
| `unit` | nothing | calculator, citation checks, refusal, retry, provider error mapping (mocked), API routing, scorers, experiment math | any real model/provider works |
| `integ` | `data/ar.pdf`, poppler | page map, extraction, table golden figures, header regressions, gold-on-page for q10/dev/val, index build/validate with a hash embedder in tmp, vset round trip | retrieval quality; the submission index |
| `live_models` | real model | BGE-M3 load, dim, revision | answer quality |
| `live_openai`, `live_ollama` | key / running Ollama | one schema-constrained call returns a valid grounded plan | quality at scale |
| `live` | `idx/` + models + provider | end-to-end answer and refusal | — |

`pytest` default = unit + integ; live tests are excluded (a skip is not a pass). Current default run on `cont`: **220 passed, 5 deselected** (the 5 live tests, not executed). Tests of note: golden B02/B03 figures; no header text in statement labels; production validator rejects the hash index and chunk-variant indexes; `check` catches a wrong gold fact; independent growth formula equals `calc`.

---

## 16. Failure handling summary

| Situation | Behaviour |
|---|---|
| No / bad API key, unknown `PROVIDER` | exit 2 before the port opens |
| `idx/` missing, corrupt, wrong model/chunking/tokenizer | exit 3 with the fix; never rebuilds |
| Models missing | exit 5 (`tcb models download` or the image) |
| Provider timeout / 5xx / rejection | answer with `status=error`, names provider+model; no other provider is tried |
| LLM returns invalid JSON | one repair retry; then claims dropped → `not_in_report`/`partial` |
| LLM cites unknown evidence id / invents a number / writes a page | that claim is dropped |
| Calculator unit/period mismatch, division by zero | "Không thể tính: …", `partial` |
| Citation outside retrieved evidence after rendering | `error` (should be unreachable) |
| Empty or punctuation-only query | `ambiguous`, no retrieval |
| Tracing I/O error | ignored; never breaks an answer |

---

## 17. Decision log (what was chosen, what was rejected, why)

| Decision | Rejected alternative | Reason |
|---|---|---|
| Half-page as citation unit | pdf page | a pdf page holds two printed pages; the grader checks printed pages |
| Poppler bbox + own parser | pdfplumber; LLM/vision extraction | deterministic, fast, no tokens; vision unneeded for text PDF |
| Coordinate-based tables with `quality` flag | trust all tables | wrong columns create wrong numbers; low tables stay text-only |
| LLM emits a plan, code renders | free-form answer + post-hoc check | page numbers and numbers can be guaranteed by construction |
| Calculator with typed ops | LLM arithmetic / generic `eval` | unit and period errors become explicit failures |
| Drop invalid claims | auto-correct them | correction by the same LLM re-opens the hallucination path |
| One repair retry | loop until valid | bounded cost and latency |
| No automatic provider fallback | silent switch to another LLM | breaks reproducibility and the offline guarantee |
| RRF (k=60) | weighted score fusion | no score calibration; RSF kept as an experiment |
| Custom tokenizer | default `bm25s` | default breaks Vietnamese numbers |
| Fail-closed index validation | warn and continue | silent degradation is worse than refusing to start |
| Eval counts verified items only | count all drafts | an unverified gold can make a correct answer "fail" or a wrong one "pass" |
| Deterministic scorers | LLM judge | reproducible, free; the cost is that fact-presence can miss a paraphrase without a numeric anchor (those items are flagged `unscorable`) |
| No reranker, no τ_low retry | BGE-reranker-v2-m3 over a pool; a score-triggered retry | one item of recall in 85 for most of the CPU time; the retry doubled retrieval on weak questions with no gain (§17.1) |
| Hash embedder only in `tests/` | use it to produce the index when BGE-M3 is unavailable | it would let a fake index pass as real |

### 17.1 Retrieval and speed options tried (measured, 13 Oct 2026)

Data: `data/dev.json` + `data/val.json`, 85 answerable items with gold pages, 19 unanswerable/ambiguous. Run `m4` on Colab (quality on a T4 GPU, rerank timing on 2 CPU cores). Recall = a gold printed page is among the blocks considered. One item = 1.2 points.

| Option | What was measured | Result | Decision |
|---|---|---|---|
| BM25 tokenizer: `tcb-vi-v1` vs bm25s default vs +bigrams (E2) | BM25-only recall | R@20: default 0.894, vi 0.882, bigram 0.894; R@5: vi 0.824, default 0.800. The only R@20 difference is one glossary item (`g-alco`) | keep `tcb-vi-v1`; a switch needs ≥ 2 pts R@20 and an index rebuild |
| BM25 vs dense vs RRF vs RSF (E3) | R@20 | BM25 0.918, dense 0.694, RRF 0.941, RSF 0.929 | keep RRF |
| RRF k ∈ {10, 30, 60, 100} (E3b) | R@20 per set | no k beats 60 by ≥ 5 pts on both sets | keep k = 60 |
| Rerank pool N × final k (E4) | recall within the k blocks given to the LLM | N10: 0.882; N20/30/40 with k ≥ 8: 0.894 (one item more). CPU rerank time grows with N (≈ 60 s per question at N = 30 on 2 cores) | pool 30 → 10, all 10 blocks to the LLM |
| Evidence count k on the rebuilt indexes (E4, no reranker) | recall of the fused top k | m6 (1,463 nodes): k 10: 0.859, 12: 0.894, 15: 0.906, 20: 0.929. m7 (final, 1,445 nodes): 5: 0.777, 8: 0.824, 10: 0.871, 12: 0.929, 15: 0.929, 20: 0.941 | rule: smallest k within 2 pts of the best. It chose 15 (m5), 20 (m6), 12 (m7): one item is 1.2 pts, so the choice moves with single items. Final k = 12 (m7) |
| BM25 vs dense vs RRF vs RSF on the final index (E3, m7) | R@20 | BM25 0.929, dense 0.718, RRF 0.941, RSF 0.953 | keep RRF (RSF is one item better, the rule needs 5 pts) |
| Glossary expansion (E5) | items lost / gained | none lost, none gained | keep (needed for abbreviations such as RBG, CASA) |
| Weak-retrieval retry τ_low (E7) | F1 of flagging unanswerable by top rerank score | τ = 0.599 (on pool 30) | not enabled: the retry is a second full search and rerank on exactly the weak and unanswerable questions, and no experiment shows it recovers a page |
| No cross-encoder at all | follows from E4: with N = k = 10 the reranker no longer selects blocks, it only orders them | recall unchanged by construction (0.882); removes ≈ 20 s per question on 2 CPU cores (val1 with the reranker at N = 10: p50 30.3 s) | **adopted 13 Oct 2026**: model, download, `tcb check` step, E4 grid and E7 removed; Docker image ≈ 2.2 GB smaller. Decided without the planned A/B eval because recall cannot change and the time saving is the whole rerank stage |
| Vector database (FAISS / Chroma / Qdrant) instead of the LlamaIndex in-memory store | time of one top-40 dense search over 1,526 × 1,024 vectors, query vector given | LlamaIndex simple store 49 ms (median), numpy 0.17 ms, FAISS flat 0.30 ms; BM25 0.14 ms | not adopted: at most 0.05 s saved per question against a rerank of tens of seconds, and a DB server adds a service to Docker |
| BGE-M3 sparse ("hybrid search" inside the model) instead of BM25 | the premise that keyword search puts the number's block at top 1 | 1 of 85 questions contains a number other than a year or date (the number is the answer, not the query); "2025" occurs in 572 of 1,526 chunks; BM25 R@1 = 0.565 | not adopted for speed: retrieval is already hybrid (BM25 + BGE-M3 dense + RRF), sparse weights come from the same forward pass so they save no time, and they need FlagEmbedding plus a re-index. A quality experiment for later |
| Larger chunks (400 to 500 words, 50 to 100 words overlap) | current chunk shape | narrative median 176 words (cap 220), table chunks are whole rows with column header and unit repeated on every row, KPI tiles 19 words, glossary 7; 0 chunks cross a printed page | not adopted: 10 blocks of ~450 words is ~2.5× the prompt (slower and dearer LLM call; the 22,000-character evidence budget would keep only ~8 blocks), blocks over ~350 words exceed the reranker's 512 tokens, and overlap across pages breaks one-page citations. Chunk size for quality (E6) not run |
| Model warm-up at start | the first question paid the model load | `Retriever.warm()` at start of chat/serve/batch/eval | adopted |

### 17.2 Answer-level results and the fixes they caused

Run `val1` (13 Oct 2026): `data/val.json` 34 items, claude-haiku-5-5, Colab 2 vCPU, still with the reranker at N = 10.

| Metric | Value |
|---|---|
| pass | 31 / 34 (0.912) |
| fact correct / page hit (answerable) | 0.893 / 0.893 |
| citation valid | 1.0 |
| refusal recall / false answers on unanswerable | 1.0 / 0.0 (5 items) |
| refusal precision | 0.625: three answerable questions refused |
| latency p50 / p95 | 30.3 s / 35.5 s |
| LLM calls per question | 1.21 (follow-up rewrites + repairs) |

All three failures were over-refusals with the gold page already in the top 4. `tcb eval` now prints, for each failed item, the plan status, the refusal reason and the check problems from its trace. Causes found:

| Item | Cause | Fix (general, not item-specific) |
|---|---|---|
| `n-hr-diff` (headcount 12.705 vs 11.848, p.52) | the calculator only accepted quantities parsed from statement tables, so any arithmetic on numbers in narrative text or KPI tiles was impossible ("diff needs 2 inputs, got 0") | **literals**: the plan may declare `L1..Ln` (evidence id, number copied verbatim, unit, period); the code accepts one only if the number occurs in that evidence block, then the normal typed calculator runs. An invented number is rejected and triggers the one repair (tests `test_calculation_on_narrative_numbers_via_literals`, `test_invented_literal_is_rejected`) |
| `sq-07` (q10 batch: RBG loan balance, KPI tile p.59) | the tile `Dư nợ vay \| 328,1` carried only the chapter name; the segment title sits on the facing page p.58 | page-title context in every node (§4.3); a paragraph that glued two tiles together is no longer linked to them |

Integer differences now print without a decimal (857, not 857,0).

---

## 18. Known gaps and limits

1. dev/val golds were checked by Claude against the PDF text (pdftotext of the cited page; calculations recomputed); a human spot-check is pending.
2. 31 low-quality note tables (53 chunks) give no table quantities; a calculation over them is possible only through literals copied from the text. A layout model could recover them (out of scope).
3. Number grounding ignores integers ≤ 12.
4. Fact-presence scoring cannot judge a purely textual answer without numeric or keyword gold; it relies on `gold_facts` being well chosen.
5. Rounded vs exact figures: the report states some values twice at different precision (p.5 vs p.297/300); gold facts use `a | b` alternatives and `alt_pages`; this is manual per item.
6. Table rows that wrap a label across a page break may be split.
7. Single-process server; one answer at a time.
8. Cost figures use list prices recorded on 10 Oct 2026 and are estimates.
9. Disagreeing figures: the prompt asks for both figures with their own citations (rule 10 of the plan prompt); no eval item has two different figures for one item and period, so this behaviour is untested.
10. The ambiguity dev items (2) are subjective; the LLM may reasonably answer "Lợi nhuận" for the only reported year.

---

## 19. Cost and latency model (to be measured, not asserted)

Per question: ≤ 1 rewrite call (follow-ups only) + 1–2 generation calls; one query embedding with BGE-M3 on CPU (`query_embed_s` in `tcb check`; `seconds.embed` in every trace) + dense search (≈ 0.05 s) + BM25 (< 1 ms). The former rerank of 30 passages took ≈ 60 s on 2 Colab cores. `tcb eval` reports p50/p95 latency and the median seconds per stage (`embed`, `dense`, `bm25`, `llm`), tokens, and a cost estimate when the model has a list price in `conf.PRICES` (otherwise null, never 0).

---

## 20. Security and privacy

Keys are read from the environment / `.env` (git-ignored; the loader never prints values and never overrides real env vars). Only the user question and retrieved evidence leave the process in online mode; offline mode sends nothing outside the compose network. The UI renders answers with `textContent` (no HTML injection); citation links open `/pdf#page=<pdf_page>`. No user data is stored except traces under `out/trace/` on the local volume.

---

## 21. Operating the project (pointers)

- Diagnose a machine: `tcb doctor`.
- Gate order on a networked machine: `tcb check` → `tcb build data/ar.pdf out/idx` → `tcb validate out/idx` → move to `idx/` → `docker compose up --build` / `./smoke.sh`.
- Evaluation order: `tcb vset check` → `tcb exp data/dev.json data/val.json` → freeze constants → `tcb eval data/val.json` → once on `data/q10.json` → fill `SUBMISSION.md` with measured numbers only.
- Exact commands and the status table live in `README.md`; the submission text lives in `SUBMISSION.md`.

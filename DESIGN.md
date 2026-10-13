# DESIGN: Vietnamese chatbot over the Techcombank Annual Report 2025


---

## 1. Goals, constraints, non-goals

**Goal.** Answer Vietnamese questions about one document (Techcombank Annual Report 2025, 196 two-page spreads) with answers that are (a) grounded in the report, (b) cited by printed page `[tr. N]`, (c) numerically correct when arithmetic is needed, and (d) refused when the report does not contain the answer.

**Fixed choices**

| Item | Decision |
|---|---|
| Embedding model | `BAAI/bge-m3`, revision pinned in `conf.EMBED_REV` |
| Reranker | none (measured and removed, §17.1) |
| Orchestration | LlamaIndex for ingestion/vector retrieval only. No agent framework |
| LLM providers | Anthropic (every reported result, `claude-haiku-5-5`), OpenAI, or a local Ollama model on the host, chosen by `PROVIDER`. **No automatic fallback between providers** |
| Depth tracks | Document Intelligence + Retrieval and reasoning; an evaluation harness serves both |
| Index | Built once, committed as `idx/`, validated fail-closed at start-up. Never rebuilt or downloaded at run time |
| Code scope | built in four days; only what one document needs, no speculative abstractions |

**Non-goals.** Multi-document support, an LLM judge, OCR of image-only content (inventoried but not read), streaming answers, authentication, multi-process serving, runtime fine-tuning.

**Reporting rules.** Results with mocked models are reported as tests of logic, never as results of a real model. The hash embedder exists only in `tests/` and writes to a tmp dir; the production validator rejects any index not built with the pinned BGE-M3 revision. Eval items count only when `verified: true`, which in this repository means machine-checked: every gold fact was found in the PDF text of its page and every calculation recomputed (`tcb vset check`); the items and the q10, test and test2 answers were also reviewed by the author. The item notes say this.

---

## 2. The document and why it is hard

Facts measured on the real PDF:

- 197 PDF pages = cover + 196 **spreads** (two printed pages side by side). The printed page number p lives on pdf page `(p+2)//2` if p is odd (right half) else `p//2+1` (left half). A citation unit is therefore a **half-page**: 393 halves.
- Printed page numbers are read from footers on 380 halves; 13 divider/blank halves are inferred from neighbours (node-level: 1,439 nodes footer-sourced, 6 inferred).
- Content kinds: narrative prose, **KPI tiles** (big number + caption), **financial-statement tables** (forms B01–B05 and notes), and a **glossary** (163 terms).
- Vietnamese numbers: `.` is the thousands separator and `,` the decimal (`1.192.344.137`, `14,6%`, negative as `(181.150)`). Units are in headers (`Triệu đồng`, `Nghìn tỷ đồng`). The same quantity appears at different scales: total assets is `1.192 nghìn tỷ đồng` on p.5 and `1.192.344.137 triệu đồng` on p.297.
- Images: 428 embedded images on 117 pages, counted in `report.json`. A visual audit (13 Oct 2026) of every raster image covering at least 1 % of a half-page found photographs, portraits, logos, backdrops and adverts; text over badges and diagrams is vector text and is extracted. Information carried only by graphics: the labels of one diagram (p.10, the "Customer Brain" labels), not read; and the up or down triangles drawn before KPI change figures. Run test1 showed the cost of the second: on p.264 the change figure was extracted without its direction and the model wrote "giảm 7%" for an up triangle. The triangles are vector paths, so they are now read without OCR (§4.1): 90 on 30 spreads, of which 63 sit just left of a figure on the same line and become a marker after it (`7% (▲) so với năm 2023`); the rest decorate charts. No OCR or vision model is used.

Failure modes that a naive "split text, embed, ask GPT" pipeline hits on this document, and where this design answers each:

| Failure | Where handled |
|---|---|
| Wrong page cited (spread vs printed page) | §4.2 page map; citation page comes from node metadata only (§8) |
| Table values scrambled / columns lost | §4.3 coordinate-based table parser + `quality` flag |
| `1.192` tokenised as `192`, `14,6%` as `14` and `6` | §4.5 tokenizer `tcb-vi-v2` |
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
    + bm25s index (tokenizer tcb-vi-v2)
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

**Drawn direction triangles** (`pdf.direction_arrows`, pdfminer.six): a filled path of 2.5 to 7 × 1.2 to 4.5 pt with exactly three corners and a horizontal base is an up triangle if the base is at the bottom, down otherwise. One placed just left of a figure on the same text line adds `(▲)` or `(▼)` after that figure. The marker is ignored when tiles are detected (it is not text weight), and the plan prompt says what it means. A marker, not the word `tăng`, because the triangle is printed next to levels too (`Tỷ lệ CASA 40,4% (▲)`), where "tăng 40,4%" would be false. Scan cost about 18 s per build.

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

Header handling (reworked after the income statement and cash flow were found mis-parsed; covered by regression tests on the real PDF):
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
- Run test1 (`x-masan-fee`) showed the note heading replacing the caption of a sub-table under it (`Thu nhập từ hoạt động dịch vụ` under note 44, related parties): the rows lost their metric and the model declined. A short caption line within 45 pt above the table, below the note heading and not band text, is now kept as the table's first row, and a chunk that starts inside a sub-table repeats that sub-table's heading row.

**Quality flag.** `quality="high"` only if every column header carries a date or `Năm YYYY`, or the table has ≤ 3 clean columns. Currently 66 high / 53 low chunks (36 of 67 tables). The primary statements (printed pages 297–302, forms B02 to B04: financial position, off-balance items, income, cash flow) are all high. Low tables (multi-column notes: provisions, segment, related-party, maturity gaps) keep their text view and are **excluded from calculator quantities** rather than guessed.

*Alternative rejected:* marking every table high and letting the LLM sort out columns; that is exactly how wrong numbers get produced from scrambled headers.

### 4.5 Tokenizer `tcb-vi-v2` (BM25)
`bm25s`' default tokenizer splits `1.192` into `192`. `vi.tokens` instead keeps dates, form codes (`B03/TCTD-HN`) and numbers whole, emits the canonical form too (`1.192.344.137` → `1192344137`, `14,6%` → `14.6%`), and emits a diacritic-folded copy of every word, so `Tăng trưởng` and `tang truong` both match. v2 differs from v1 only in canonical decimals (`16,0` → `16`). On the final index BM25 alone reaches R@20 0.929 against 0.718 for dense alone (E3, run m7).

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
1. **BM25** over `tcb-vi-v2` tokens → top 40 (`TOP_BM25`).
2. **Dense** (BGE-M3 via LlamaIndex) → top 40 (`TOP_DENSE`).
3. **KPI label match** (`find.label_matches`): a tile whose whole label (3 or more syllables, no figure; 94 labels) occurs in the question forms a third ranking, longest label first. Added after test1 `x-credcost`: the tile `Chi phí tín dụng | 0,6%` is short, so it ranked below longer statement rows on provision expense. E8 measures it.
4. **RRF** with k = 60 (`RRF_K`) over the two or three rankings; the top **12** (`FINAL_EVIDENCE`, E4 run m7) go to the LLM, further limited by a 22,000-character prompt budget (`PROMPT_EVIDENCE_CHARS`) so prompts stay within the local model's context (`OLLAMA_NUM_CTX`).

There is no cross-encoder and no score threshold (§17.1). `Result` keeps every stage's ranking (`bm25`, `dense`, `fused`, `ranked`) and the seconds spent on BM25, query embedding and dense search, so evaluation can say *which stage lost the gold page* and *where the time went* (§14).

### 6.2 Glossary expansion
If the query contains an upper-case glossary abbreviation (e.g. CASA), the report's own definition (first 100 characters) is appended in parentheses: `CASA (Tiền gửi không kỳ hạn…)`. The expanded query is used by both stages. E5 (run m4): no item lost, none gained; kept because abbreviations such as RBG and CASA need it on unseen questions.

### 6.3 No weak-retrieval retry
An earlier design retried a query once when the top cross-encoder score fell below τ_low (E7). E7 proposed τ = 0.599, but the retry meant a second full search on exactly the weak and unanswerable questions, and no experiment showed it recovering a page. It was removed together with the reranker; refusal relies on the plan and the validators (§10).

### 6.4 Why this shape
- BM25 + dense because figures/abbreviations need lexical exactness while paraphrased narrative needs semantics.
- RRF because it needs no score calibration between BM25 and cosine; k = 60 is the literature default and is only changed if E3b shows ≥ 5 points of R@20 on **every** set.
- No reranker (E4, run m4: dev+val, 85 answerable items): a cross-encoder over 20 to 40 candidates found a gold page for 0.894 of items, the fused top 10 for 0.882, i.e. one item, at about 3 s of CPU per candidate pair on 2 cores (≈ 60 s per question at 20, ≈ 90 s at 30; 5 questions each). Over the fused top 10 that were then given to the LLM it could only reorder them.
- Tokenizer stays `vi`: E2 favoured `default` by one item at R@20 (0.894 vs 0.882 in run m4; 0.941 vs 0.929 in m7) while `vi` is ahead at R@5/R@8, so no rebuild.
- One retrieval path: no HyDE, multi-query or agents; each adds LLM calls and a failure surface.

---

## 7. Generation: the plan

### 7.1 Plan schema (`plan.py`)
The LLM returns **one JSON object**:

```
status: answered | partial | not_in_report | ambiguous
sub_questions: [str]      # prompt guidance only: the code does not check that every part was answered
claims: [{text_vi, evidence:[E#], calc_refs:[C#]}]
calculations: [{cid, op, inputs:[Q#], to?}]     op ∈ growth_pct diff ratio share_pct cagr sum convert
clarification_vi: str?
```
Evidence blocks are `[E1]…[E12]`; extractable numbers from high-quality tables are offered as `[Q1]…[Qn]` (`label | period | value unit`, capped at 60). Claim text may contain `{C1}` placeholders where a calculator result is inserted. The system prompt (Vietnamese) forbids outside knowledge, requires numbers copied verbatim in Vietnamese format, forbids writing page numbers, forbids doing arithmetic, and tells the model to choose `not_in_report` rather than guess.

### 7.2 Providers (`llm/`)
| Provider | Structured output | Notes |
|---|---|---|
| OpenAI | `response_format: json_schema` | SDK retries off (`max_retries=0`) so retry policy is ours |
| Anthropic | forced `tool_use` with the schema | |
| Ollama | `format=<schema>`, `num_ctx` 32768 (`OLLAMA_NUM_CTX`) | default `qwen2.5:7b`; the measured prompts (median ~10k, max ~15k tokens in the final runs) fit |

Errors are typed: `ConfigError` (exit 2), `ProviderTimeout`, `ProviderUnavailable`, `ProviderRejected`, `GenerationError`; every message names provider and model. An `error` answer is returned to the user as such. **No other provider is ever tried.** Token usage, latency and a list-price cost estimate (`conf.PRICES`) are recorded per call.

### 7.3 One repair retry (`gen.py`)
If the reply is not parseable JSON for the schema, or `cite.check` finds problems, the same provider is called **once more** with the previous output and a short list of the specific errors. Whatever is still invalid after that is dropped by the pipeline (§8), not retried. Bound: ≤ 2 generation calls + ≤ 1 rewrite call per question.

---

## 8. Deterministic validation and citation (`cite.py`)

`cite.check(plan, evidence, quantities, question)` per claim:
1. Every `evidence` id exists among the retrieved blocks and has a printed page.
2. Every `{C#}`/`calc_refs` id exists among the plan's calculations; calculation inputs exist among the extracted quantities.
3. The text contains **no** page reference (`tr. 5`, `trang 5` …): pages are never LLM output.
4. **Every number in the claim text must appear in the cited evidence** (or in the raw cells of quantities used by the claim's calculations, or in the question), compared in canonical form only, so `14,6%` ≡ `14.6`. Raw forms are not compared: the raw `179.501` (thousands) is the canonical form of `179,501` (179.5), and comparing both let a claim `179,501 nghìn tỷ` pass against evidence `179.501 tỷ` (found in run dev9, fixed and tested). Integers ≤ 12 are exempt (list counts, months), a documented gap.
5. **Magnitude.** A number printed with a magnitude in the cited evidence (`1.192 | Nghìn tỷ đồng`, `18 triệu`, `40,4%`, or the unit of a table column, `31/12/2025 Triệu đồng: 1.192.344.137`) must carry the same magnitude in the claim. `1.192 tỷ đồng` or a bare `1.192` fails although every digit matches: that is the silent factor-of-1,000 error. `điểm phần trăm` has the magnitude of `%`, because the report prints percentage-point changes with `%` (`1,9% (▼) so với 32,7%`); before this, a correct `giảm 1,9 điểm phần trăm` was dropped (dev13 `n-cir-diff`).
6. At least one evidence id or calculation per claim.
7. **Sign, both ways.** A figure written negative must be printed negative; a figure the evidence prints only negative (`(100)`) may be written positive only with a word that carries the sign (`lỗ`, `giảm`, `chi phí`, `dự phòng` …).
8. **Copied figures (literals).** The number must be printed in the cited block with the same sign, the currency and magnitude printed next to it (or, when none is, present in the block), and the declared year must be the year it is printed under: the last year before it in the same sentence or table cell (`Năm 2025 Triệu đồng: 3.503; Năm 2024 …`), else the first after it when no other figure comes in between (`30,8% | 1,9% so với 32,7% trong năm 2024`: 2024 belongs to 32,7%); `năm trước` / `cùng kỳ` right after it means the year before. Only a block that prints no year lets the question's year stand in.
9. **Restated amounts.** After calculation results are inserted, an amount given as `X, tương đương Y` must equal Y within rounding (dev13 `n-fu-eq` paired a sum of 834.241.663 triệu đồng with 776,5 nghìn tỷ).

These checks tie a figure to its block, not to a row or a year inside a table block: `978.798.549` (the 2024 total) stated as the 2025 total of the same statement block passes. The citation is what lets the analyst see the row; tying claim figures to rows is a next step (§18).

**Restated figures.** Run `q10e` showed 4 of 10 answers restating a figure from a second page. The prompt now asks for one claim per figure listing every evidence id (rule 9), and the code flags in the trace (`restated_figures`) any claim whose figures were all stated by an earlier claim (same number, or the same figure at another rounding or unit within half a unit of the rounded one). Removing such claims was tried (runs dev9/val9) and dropped: both claims it removed on dev+val carried a remark the earlier claim did not ("dẫn đầu hệ thống ngân hàng"), while the prompt alone left no restatement in q10.

A claim failing any check is **dropped**, not patched. Rendering removes an amount restated as itself (`27.538 tỷ đồng, tương đương 27.538 tỷ đồng`) and remarks on where the report places a figure (`… được báo cáo tóm tắt ở phần điểm nhấn`). A drawn-arrow marker (`(▲)`, §4.1) is removed from a rendered sentence that already has `tăng` or `giảm`, and kept where it is the only sign of direction. Rendering then joins the kept claims into one paragraph, each sentence ending `... [tr. N].` where N is the `printed_page_number` of the first block the model lists for the sentence plus the blocks that print the most of its figures (`cite.source_ids`; all cited blocks for a sentence without figures), merged and sorted. Every listed page was shown before (test2: up to five for one figure); the figure blocks alone dropped legitimate pages the model had put first (run 904516e: three items) (the prompt asks for a first sentence with its subject and follow-on sentences that do not repeat it, one claim per indicator with its change and the remark the report prints next to it); calculator results use the pages of their input quantities. A final `verify_rendered` pass asserts every page in the rendered answer belongs to retrieved evidence (defence in depth; failure → `error` status).

The answer payload also lists `citations` (eid, page, pdf_page, section, chunk_id, snippet) for the UI to open the source.

---

## 9. Calculator (`num.py`, `calc.py`)

`num.from_node` converts cells of **high-quality tables only** into `Quantity(value: Decimal, dim, scale, period, page, raw)`; units/periods come from the column header (e.g. `Năm 2025 Triệu đồng` → VND, 10⁶, year 2025). Seven typed operations, all `Decimal`, all strict. Every input is a table quantity or a checked literal; a calculation cannot take another calculation's result (`C1` as an input of `C2` fails the check and the claim is dropped), so each result is one step from printed figures. A `sum` or `diff` can take `to` (`nghìn tỷ đồng`) to rescale its result in the same step; a rescaled amount is shown without trailing zero decimals:

| op | Rule |
|---|---|
| `growth_pct(curr, prev)` | same dim & period kind, curr later than prev, prev ≠ 0 → `%` |
| `diff(a, b)` | same dim & kind; `% − %` → percentage points |
| `ratio(n, d)` | same dim, d ≠ 0 → `lần` |
| `share_pct(part, total)` | same dim and same period, total ≠ 0 |
| `cagr(start, end)` | years ≥ 1, start > 0, end ≥ 0 |
| `sum(qs)` | same dim and period |
| `convert(q, to)` | VND/USD rescale only (`triệu đồng` → `nghìn tỷ đồng`) |

Mismatches raise typed errors (`UnitMismatch`, `PeriodMismatch`, `ZeroDenominator`, `InvalidInput`), which the pipeline turns into "Không thể tính: …" notes and a `partial` status, never a guessed number. Display is Vietnamese (`21,8%`, `1.192.344.137`). Golden test on the real PDF: total assets growth from p.297 cells = `21,8%`, equal to the report's own p.5 KPI; income-statement growth `13,6%` (total operating income) and `18,2%` (profit before tax).

*Alternative rejected:* letting the LLM compute and then checking it: a check that recomputes is just the calculator with extra steps.

---

## 10. Refusal policy and statuses (`refuse.py`)

| Status | Meaning |
|---|---|
| `answered` | all claims kept, no calc errors, no validation leftovers |
| `partial` | some claims/calcs dropped or LLM said partial; answer plus a note on what could not be verified |
| `not_in_report` | LLM declined ("the report has no information"), or **no** claim survived validation (fail closed; the message then says the figures could not be verified, not that the report lacks them) |
| `ambiguous` | the question lacks entity/period; returns a clarifying question |
| `error` | provider/index/retrieval failure; message names provider/model |

Fail-closed means an answer exists only through claims that survived §8. Retrieval-score thresholds never suppress a grounded answer on their own.

---

## 11. Multi-turn (`chat.py`)

A question is treated as a follow-up only if there is history **and** it contains a word that points back (`còn`, `đó`, `này`, `năm trước` …) or is a fragment of four words or fewer (`Năm 2024?`); a short question that names its own subject is asked as it is. Batch files carry context the same way, through a `history` list or a shared `session` id. One LLM call (JSON `standalone`) rewrites it using the last two turns. The rewrite is accepted only if every number in the user's question is still present in it (faithfulness guard); otherwise the deterministic fallback `"<previous standalone> — <question>"` is used. The mode (`none | llm | concat`) is recorded in the trace. A provider error during rewrite is **not** swallowed: it surfaces as an `error` answer (no silent fallback). History is *context for understanding the question, never evidence*: an earlier answer cannot be cited. Sessions live in memory (cap 200) keyed by `session_id`.

---

## 12. Surfaces

- **HTTP (`api.py`, FastAPI):** `GET /health`, `POST /chat {question, session_id?}`, `GET /evidence/{chunk_id}`, `GET /pdf`, `GET /` (single-file UI `ui.html`), `/docs`. Single process, one answer at a time under a lock. Status codes map error kinds (config/timeout/unavailable/rejected). Every answer's full trace is appended to `out/trace/chat.jsonl`; tracing failures never break an answer.
- **CLI (`tcb`):** `doctor` (diagnose this machine), `models download`, `check` (gate: load BGE-M3, related text closer than unrelated, one query embedding timed, log RSS), `build`, `validate`, `ask`, `chat`, `batch`, `eval` (`--compare`, `--include-unverified`), `exp` (retrieval experiments), `vset check` (gold facts on gold pages), `smoke` (Ollama), `serve`. Exit codes: 2 config, 3 index, 4 error answer, 5 model missing, 6 smoke.
- **Boot order (`boot.py`):** read config → validate index → load models → serve. A bad index exits 3 before models are touched; a missing key exits 2 before the port opens.

---

## 13. Packaging and portability

- **Docker (`Dockerfile`, `compose.yaml`):** `docker compose up --build` is the online default. The image installs CPU torch from the PyTorch index first, then `pip install --require-hashes --no-deps -r requirements.lock` (hash-pinned, Python 3.12 linux-x86_64, no torch/CUDA); BGE-M3 is downloaded at build time at pinned revisions into `/models` with a `.rev` marker. `idx/` and `data/` are mounted, not baked, so the index can be replaced without rebuilding the image. Layers are kept lean on purpose: `COPY --chown` instead of `RUN chown -R` and no in-image weight conversion, because each would duplicate 2.3 GB of weights in a new layer (a first Docker build on a 32 GB Codespace ran out of disk exporting layers). The one-off index build runs on Colab (`TCB_DEVICE=cuda` only speeds up embedding; model and revision are the pinned ones), not inside Docker.
- **Local model:** Ollama runs on the host (`ollama pull qwen2.5:7b`); `PROVIDER=ollama` and the default `OLLAMA_BASE_URL=http://host.docker.internal:11434` reach it from the container (`extra_hosts: host-gateway` in `compose.yaml`). `tcb smoke` sends it one fixed question and reports the latency and whether the answer passed the checks. A local run was planned for the evaluation as well; on the machines available (2 vCPU, no GPU) a 7B model needs minutes per answer at these prompt sizes, so the reported results use claude-haiku-5-5 and the adapter is tested against a mocked server. An earlier Compose profile that ran Ollama in its own container was dropped in favour of the host setup, which needs no extra service.
- **Local runs (no Docker):** `pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu && pip install -e ".[dev]"`, `python -m tcb.models download`; `.env` loader (does not override real env vars); `TCB_MODELS_DIR`, `POPPLER_BIN`, `PYTHONUTF8`; `import resource` guarded for Windows; explicit UTF-8 everywhere. `tcb doctor` prints a one-screen diagnosis (python, packages, poppler, PDF hash, models, index, keys, RAM/disk, Docker) with a fix per failure.
- **Locks:** `lock.sh` produces `requirements.lock` and fails if torch/CUDA leak in.

---

## 14. Evaluation design

### 14.1 Datasets
| File | Content | Rule |
|---|---|---|
| `data/q10.json` | the 10 published sample questions (9 answerable, 1 not) | used to debug after batch runs (sq-07 led to page-title context, sq-03/sq-07 to the magnitude fixes); fixes are general and tested on other pages, but q10 is not an independent benchmark; additively annotated with `gold_facts`, `alt_pages` (e.g. 297 for total assets), `type`, `verified` |
| `data/dev.json`, `data/val.json` | 70 + 34 items written by hand (statement tables, KPI tiles, narrative, segments, subsidiary, governance, investor relations, glossary, calculations, follow-ups, 19 unanswerable or ambiguous) | gold machine-checked against the PDF text of the cited page (`tcb vset check`), calculations recomputed; reviewed by the author. One page added after review: `n-equity` alt page 256 (the same equity printed as 179.501 tỷ đồng), rescored |
| `data/test.json` | 22 items written after the last code change, on pages no other set uses, run once (test1) | its three failures led to the fixes in §4.1, §4.4 and §6.1, so after them it is no longer independent |
| `data/test2.json` | 22 items written after those fixes, on pages no set above uses, frozen before its first run | the independent estimate (first run on `6d84763`); disjointness enforced by a test |

Unanswerable topics were chosen only if a full-text search of the report found no mention (Vietcombank, Campuchia, Bitcoin, Momo, Nasdaq, World Cup, 2027, MB Bank, Moody's; enforced by a test). Topics that **are** in the report (dividends, 2029/2030, remuneration, 2026 plan) are not used as "unanswerable". `tcb vset check` asserts every answerable item's gold facts are present, as text, on its gold pages (for calc items: the *inputs* are). The first table items were drafted by a generator from the statements; it and its CSV review loop were removed once the sets were checked, so they cannot be regenerated over by accident.

Item types: `single`, `multi`, `calc`, `followup`, `refusal`; `expect_status: ambiguous` marks questions whose right behaviour is a clarifying question; these are excluded from refusal precision/recall.

### 14.2 Scorers (deterministic, no LLM judge)
- **Fact present:** every `gold_facts` entry must appear in the answer; `a | b` = either alternative; numbers compared in canonical form and with the magnitude the gold prints (`100 tỷ` does not satisfy `100 triệu`). The gold figure stated more precisely or in another unit also counts (`53.391 tỷ` for `53,4 nghìn tỷ`, within half a unit of the gold's last digit); a less precise one does not (`40%` for `40,4%`). Re-scoring runs dev11/val11/q10i with this rule changed two items (sq-06, n-equity), both correct answers stated in another unit. After test2 the rule was extended to round counts without a magnitude: a gold ending in three or more zeros is rounded at its last non-zero digit (`hơn 303.000` is met by `303.446`, `1.000.000` by `1 triệu`); re-scoring the five final runs changed two items to pass (test2 `y-visaeco`, val `n-tinchap`) and nothing else.
- **Citation valid:** every cited page belongs to the evidence retrieved for that question.
- **Page hit / precision:** cited pages vs `gold_printed_pages ∪ alt_pages`, widened at scoring time by `widen_gold` to every page with one node that prints all the numeric gold facts (only for facts with a distinctive number: a decimal or 4+ digits). `page_hit_declared` counts the written pages only; SUBMISSION reports both.
- **Refusal:** precision and recall over unanswerable items, plus false-answer rate.
- **Cost/latency:** per question from traces (p50/p95, tokens, estimated USD).

### 14.3 Stage attribution
For each failed item, the first failing stage in this order: `system` (provider/index error) → `retrieval:recall_miss` (gold page in neither BM25 nor dense top-40) → `retrieval:fusion_drop` → `retrieval:rank_drop` (gold page ranked below the `FINAL_EVIDENCE` blocks given to the LLM) → `citation` (invalid/wrong page) → `calc` → `generation:wrong_or_missing_fact` → `refusal:over_refusal` / `under_refusal`. This is what makes "what did the numbers make you change" answerable.

### 14.4 Run artifacts
`tcb eval` writes `traces.jsonl` (full per-question trace), `results.jsonl`, `summary.json` and `config.json` (set sha256, provider, model, index identity, evidence count, RRF k, device, and **`submission_grade`**: true only with the real BGE-M3 index). `tcb eval --compare A B` reports fixed / regressed / verdict-changed items and metric deltas, and warns if the two runs differ in set, provider, model, τ, k or index. `tcb eval --rescore RUN` scores a finished run again with the current scorer, reusing the saved answers and frozen gold items (no model call); the copy records `rescored_from` and the scorer's commit.

### 14.5 Experiments (`tcb exp`, retrieval only, zero token cost)
| ID | Question | Decision rule |
|---|---|---|
| E2 | tokenizer `tcb-vi` (production) vs default vs bigram | switch only with ≥ 2 pts R@20 (a switch needs an index rebuild) |
| E3 / E3b | BM25 vs dense vs RRF vs RSF; RRF k sweep | keep k = 60 unless ≥ 5 pts R@20 on **every** set |
| E4 | how many fused blocks go to the LLM (k = 5 … 20) | smallest k within 2 pts of the best recall (run m4 measured the former reranker grid N × k instead; §17.1) |
| E5 | glossary expansion | keep only if no regression |
| E6 | narrative chunk size | needs index rebuild; experimental indexes are rejected by `serve` |
| E8 | KPI label ranking on or off | keep if no gold page leaves the final top 12 |

The runner *suggests* a decision by these rules and never edits `conf.py`. Order of work: experiments on dev+val → freeze → eval on val and q10 → `test.json` written after the last code change and run once (test1) → general fixes for its three failure modes → `test2.json` written and frozen, run once. Val was also used to find failures and to choose k, so it is a validation set, not a held-out set; q10 was used to debug too (see §14.1). Neither measures generalisation; the graders' held-out set does.

---

## 15. Test strategy: what each category proves

| Marker | Needs | Proves | Does not prove |
|---|---|---|---|
| `unit` | nothing | calculator, citation checks, refusal, retry, provider error mapping (mocked), API routing, scorers, experiment math | any real model/provider works |
| `integ` | `data/ar.pdf`, poppler | page map, extraction, table golden figures, header regressions, gold-on-page for q10/dev/val, index build/validate with a hash embedder in tmp, vset round trip | retrieval quality; the submission index |
| `live_models` | real model | BGE-M3 load, dim, revision | answer quality |
| `live_anthropic`, `live_openai`, `live_ollama` | key / running Ollama | one schema-constrained call returns a valid grounded plan | quality at scale |
| `live` | `idx/` + models + provider | end-to-end answer and refusal | |

`pytest` runs unit and integration tests by default, and all of them pass; the five live tests need keys or running services and run on demand (`pytest -m live_anthropic`, …). Tests of note: golden B02/B03 figures; no header text in statement labels; production validator rejects the hash index and chunk-variant indexes; `check` catches a wrong gold fact; independent growth formula equals `calc`.

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
| No automatic provider fallback | silent switch to another LLM | breaks reproducibility: every answer names the provider and model that produced it |
| RRF (k=60) | weighted score fusion | no score calibration; RSF kept as an experiment |
| Custom tokenizer | default `bm25s` | default breaks Vietnamese numbers |
| Fail-closed index validation | warn and continue | silent degradation is worse than refusing to start |
| Eval counts verified items only | count all drafts | an unverified gold can make a correct answer "fail" or a wrong one "pass" |
| Deterministic scorers | LLM judge | reproducible, free; the cost is that fact-presence can miss a paraphrase without a numeric anchor (those items are flagged `unscorable`) |
| No reranker, no τ_low retry | BGE-reranker-v2-m3 over a pool; a score-triggered retry | one item of recall in 85 for most of the CPU time; the retry doubled retrieval on weak questions with no gain (§17.1) |
| Hash embedder only in `tests/` | use it to produce the index when BGE-M3 is unavailable | it would let a fake index pass as real |

### 17.1 Retrieval and speed options tried (measured 11 to 13 Oct 2026)

Data: `data/dev.json` + `data/val.json`, 85 answerable items with gold pages, 19 unanswerable/ambiguous. Run `m4` on Colab (quality on a T4 GPU, rerank timing on 2 CPU cores). Recall = a gold printed page is among the blocks considered. One item = 1.2 points.

| Option | What was measured | Result | Decision |
|---|---|---|---|
| BM25 tokenizer: `tcb-vi-v1` vs bm25s default vs +bigrams (E2) | BM25-only recall | R@20: default 0.894, vi 0.882, bigram 0.894; R@5: vi 0.824, default 0.800. The only R@20 difference is one glossary item (`g-alco`) | keep `vi` (v1 then; v2 differs only in canonical decimals); a switch needs ≥ 2 pts R@20 and an index rebuild |
| BM25 vs dense vs RRF vs RSF (E3) | R@20 | BM25 0.918, dense 0.694, RRF 0.941, RSF 0.929 | keep RRF |
| RRF k ∈ {10, 30, 60, 100} (E3b) | R@20 per set | no k beats 60 by ≥ 5 pts on both sets | keep k = 60 |
| Rerank pool N × final k (E4) | recall within the k blocks given to the LLM | N10: 0.882; N20/30/40 with k ≥ 8: 0.894 (one item more). CPU rerank time grows with N (≈ 60 s per question at N = 20, ≈ 90 s at N = 30, on 2 cores) | pool 30 → 10, all 10 blocks to the LLM |
| Evidence count k on the rebuilt indexes (E4, no reranker) | recall of the fused top k | m6 (1,463 nodes): k 10: 0.859, 12: 0.894, 15: 0.906, 20: 0.929. m7 (1,445 nodes): 5: 0.777, 8: 0.824, 10: 0.871, 12: 0.929, 15: 0.929, 20: 0.941. m8 (final index, KPI label ranking on): 5: 0.788, 8: 0.847, 10: 0.882, 12: 0.929, 15: 0.941, 20: 0.953 | rule: smallest k within 2 pts of the best. It chose 15 (m5), 20 (m6), 12 (m7), 15 (m8): one item is 1.2 pts, so the choice moves with single items. k = 12 was frozen before test2 and kept |
| KPI label ranking on or off (E8, m8) | gold page in the final 12 | no gain and no loss on dev+val (no question there names a short tile that was ranked out); it is what brought test1 `x-credcost` into the evidence (test1b) | keep |
| BM25 vs dense vs RRF vs RSF on the final index (E3, m7) | R@20 | BM25 0.929, dense 0.718, RRF 0.941, RSF 0.953 | keep RRF (RSF is one item better, the rule needs 5 pts) |
| Glossary expansion (E5) | items lost / gained | m4 to m7: R@8 unchanged; m4: one item lower at R@5 with expansion; m8: one item higher at R@8 (0.929 vs 0.918) | keep (needed for abbreviations such as RBG, CASA) |
| Weak-retrieval retry τ_low (E7) | F1 of flagging unanswerable by top rerank score | τ = 0.599 (on pool 30) | not enabled: the retry is a second full search and rerank on exactly the weak and unanswerable questions, and no experiment shows it recovers a page |
| No cross-encoder at all | follows from E4: with N = k = 10 the reranker no longer selects blocks, it only orders them | recall unchanged by construction (0.882); removes ≈ 20 s per question on 2 CPU cores (val1 with the reranker at N = 10: p50 30.3 s) | **adopted 13 Oct 2026**: model, download, `tcb check` step, E4 grid and E7 removed; Docker image ≈ 2.2 GB smaller. Decided without an answer-level A/B run, because recall cannot change and the time saving is the whole rerank stage |
| Vector database (FAISS / Chroma / Qdrant) instead of the LlamaIndex in-memory store | time of one top-40 dense search over 1,526 × 1,024 vectors (an earlier index), query vector given | LlamaIndex simple store 49 ms (median), numpy 0.17 ms, FAISS flat 0.30 ms; BM25 0.14 ms | not adopted: at most 0.05 s saved per question against a rerank of tens of seconds, and a DB server adds a service to Docker |
| BGE-M3 sparse ("hybrid search" inside the model) instead of BM25 | the premise that keyword search puts the number's block at top 1 | 1 of 85 questions contains a number other than a year or date (the number is the answer, not the query); "2025" occurs in 572 of 1,526 chunks; BM25 R@1 = 0.565 | not adopted for speed: retrieval is already hybrid (BM25 + BGE-M3 dense + RRF), sparse weights come from the same forward pass so they save no time, and they need FlagEmbedding plus a re-index. A quality experiment for later |
| Larger chunks (400 to 500 words, 50 to 100 words overlap) | current chunk shape | narrative median 176 words (cap 220), table chunks are whole rows with column header and unit repeated on every row, KPI tiles 19 words, glossary 7; 0 chunks cross a printed page | not adopted: blocks of ~450 words make the prompt ~2.5× longer (slower and dearer LLM call; the 22,000-character evidence budget would keep only ~8 blocks), blocks over ~350 words exceed the 512 tokens embedded per block (`max_length`), and overlap across pages breaks one-page citations. Chunk size for quality (E6) not run |
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

### 17.3 Final runs (commit `6d84763`, 13 Oct 2026)

| Run | Pass (first scored) | Failures, first failing stage |
|---|---|---|
| test2 | 22/22 (21/22) | none after rescoring; `y-visaeco` was a scorer false negative ("hơn 1 triệu" for `1.000.000`) |
| test1b | 22/22 | none; the three test1 failure modes are fixed (not independent) |
| q10k | 10/10 | none |
| val13 | 33/34 (32/34) | `f-04`: a follow-up answered from p.257 with the same profit rounded to tỷ; the statement page ranked outside the 12 blocks |
| dev13 | 68/70 | `n-mktcap`: 247,3 nghìn tỷ for 247.310 tỷ, right value and page, a miss under the precision rule; `n-cir-diff`: a correct percentage-point answer dropped by the magnitude check, fixed after the run (§8) |

The answers also cite every page that repeats a figure (page precision 0.72 to 0.87) and, on test2, repeated the arrow marker after a verb (`tăng 24% (▲)`). The magnitude change (§8) was replayed on the 143 saved sentences of the five runs: only `n-cir-diff` changes. The marker change touches no figure or page.

**Run 904516e (after the review fixes):** dev14 66/70, val14 32/34, q10 10/10 in three samples (q10l, q10m, q10n), test2 21/22 and 22/22 in two more samples (test2b, test2c); summaries in `runs/history/`. Reading the misses: citing only the blocks that print the most figures dropped pages the model had listed first (dev `n-nps`, val `n-tinchap`, test2 `y-community`: correct figure, page outside the gold list), so the first listed block is kept as well; the year rule took the year of the comparison figure that follows (`n-cir-diff`), so a following year now counts only with no figure in between; the model summed statement rows where the report prints the total and then chained a unit conversion onto the sum (`n-fu-eq`), so the prompt says to use a printed total and `sum` takes `to`; a conversion was shown as `27.538.000,000 triệu đồng` (`f-04`). Each change was replayed on the saved plans of these runs and has a test. `n-mktcap` (247,3 nghìn tỷ for a question in tỷ) and `f-04` (gold page ranked outside the 12 blocks) remain.

**Final runs (commit `86d5252`, code as `4176c22`):** dev15 68/70, val15 31/34, q10 10/10 in three runs (q10o, q10p, q10q), test2 21/22 in two runs (test2d, test2e). The `to` added to `sum` and `diff` rejected count units (`lượt theo dõi`, `người`): test2 `y-followers` in both runs and val `n-hr-diff` were refused. Fixed afterwards (`to` rescales money only and is otherwise ignored, test `test_to_on_a_count_sum_is_ignored`); replaying the saved plans gives 264.000 and 857 and changes no other saved calculation. Every miss of the final dev and val runs:

| Set | Item | Question | Answer given | Cause | Status |
|---|---|---|---|---|---|
| dev | `n-mktcap` | Market capitalisation at 31/12/2025, in tỷ đồng | "247,3 nghìn tỷ đồng" [tr. 38] | right figure and page, rounded; the page prints 247.310 tỷ and the prompt rule on the asked unit did not hold | misses in every run; a code check for the asked unit is the next step |
| dev | `t-18` | Fixed assets at 31/12/2025 | the two components, 6.343.732 and 5.779.202 triệu đồng [tr. 297] | the total 12.122.934 was not stated | varies between runs (earlier runs summed them) |
| val | `n-hr-diff` | Headcount increase 2025 over 2024 | refused | the calculator bug above | fixed; the replayed calculation gives 857 |
| val | `f-04` | Follow-up "Còn năm 2024 thì sao?" (profit before tax) | "27.538 tỷ đồng, tương đương 27.538.000 triệu đồng" [tr. 257] | right value, rounded; the statement page (p.300, 27.538.368 triệu) ranked outside the 12 blocks | misses in every run |
| val | `a-year` | "Lợi nhuận là bao nhiêu?" (ambiguous) | profit before and after tax for 2025 [tr. 5] | answered instead of asking which profit and year | varies between runs |

None of them cites a page outside its evidence or states a figure the report does not print.

---

## 18. Known gaps and limits

1. Gold facts are checked against the PDF text of their page by code; the wording of a gold answer was reviewed by the author, not by a panel of analysts.
2. 31 low-quality note tables (53 chunks) expose no automatically parsed quantities; a calculation over them is possible only through checked literals copied from the text. A layout model could recover them (out of scope).
3. Number grounding exempts integers up to 12 that carry no unit, percent or currency (list counts such as `3 yếu tố`); `5 tỷ đồng` is checked.
4. Fact-presence scoring cannot judge a purely textual answer without numeric or keyword gold; it relies on `gold_facts` being well chosen.
5. Rounded vs exact figures: the scorer accepts the gold figure stated more precisely or in another unit (§14.2). The report states some values twice at different precision (p.5 vs p.297/300); gold facts use `a | b` alternatives and `alt_pages`; this is manual per item.
6. Table rows that wrap a label across a page break may be split.
7. Single-process server; one answer at a time.
8. Cost figures use list prices recorded on 10 Oct 2026 and are estimates.
9. Claims without numbers are not checked for content: a sentence citing a page is only as true as the model makes it (numbers, units, signs, pages and calculations are checked; words are not).
10. Disagreeing figures: the prompt asks for both figures with their own citations (rule 10 of the plan prompt); no eval item has two different figures for one item and period, so this behaviour is untested.
11. Direction triangles are read only when drawn as a three-corner path next to a figure (§4.1); a direction drawn any other way (an arrow icon, a coloured bar) is still lost.
12. The KPI label ranking needs the question to name the tile's whole label; a paraphrase (`chi phí rủi ro tín dụng`) relies on BM25 and dense alone.
13. Rule 7 of the plan prompt (no substitute metric; a total is summed from its components) is a prompt rule, not a code check; evaluation catches a violation only through the gold figure.
14. The ambiguity dev items (2) are subjective; the LLM may reasonably answer "Lợi nhuận" for the only reported year.
15. A claim's figure is tied to its block, not to the row or column inside a table block (§8); a wrong-row figure from the right table passes the check and is caught only by reading the citation.
16. Dense vectors are computed on the first 512 tokens of a node (`max_length=512`); 13 table chunks exceed about 1,500 characters (up to 3,556), roughly that limit, so their last rows may be found by BM25 only. Splitting tables by tokens needs an index rebuild.
17. Every reported number is one run at the provider's default sampling; variance across runs is estimated only where a set was run more than once.

---

## 19. Cost and latency model (measured values: SUBMISSION and `runs/`)

Per question: ≤ 1 rewrite call (follow-ups only) + 1–2 generation calls; one query embedding with BGE-M3 on CPU (`query_embed_s` in `tcb check`; `seconds.embed` in every trace) + dense search (≈ 0.05 s) + BM25 (< 1 ms). The former rerank of 30 passages took ≈ 60 s on 2 Colab cores. `tcb eval` reports p50/p95 latency and the median seconds per stage (`embed`, `dense`, `bm25`, `llm`), tokens, and a cost estimate when the model has a list price in `conf.PRICES` (otherwise null, never 0).

---

## 20. Security and privacy

Keys are read from the environment / `.env` (git-ignored; the loader never prints values and never overrides real env vars). In online mode the provider receives the question, the retrieved evidence, the table quantities and, for follow-ups, the last two questions with the first 300 characters of their answers; with `PROVIDER=ollama` nothing leaves the machine. The UI renders answers with `textContent` (no HTML injection); citation links open `/pdf#page=<pdf_page>`. No user data is stored except traces under `out/trace/` on the local volume.

---

## 21. Operating the project (pointers)

- Diagnose a machine: `tcb doctor`.
- Gate order on a networked machine: `tcb check` → `tcb build data/ar.pdf out/idx` → `tcb validate out/idx` → move to `idx/` → `docker compose up --build` / `./smoke.sh`.
- Evaluation order: `tcb vset check` → `tcb exp data/dev.json data/val.json` → freeze constants → `tcb eval data/val.json` → once on `data/q10.json` → fill `SUBMISSION.md` with measured numbers only.
- Exact commands and the status table live in `README.md`; the submission text lives in `SUBMISSION.md`.

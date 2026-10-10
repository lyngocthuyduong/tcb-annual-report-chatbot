# tcb — Vietnamese chatbot over the Techcombank Annual Report 2025

Evidence-first: every answer cites printed pages `[tr. N]`; refuses when the report lacks the answer.
Status: **M0–M2 done** (ingestion + index build/validate). Retrieval, LLM, chat, eval: next steps.

## Prerequisites
Docker (Compose v2), ≥ 8 GB RAM for Docker, ~10 GB disk, internet for the first build (PyPI + Hugging Face).

## Commands
```bash
cp .env.example .env                                       # OPENAI_API_KEY etc. (needed from M4)
docker compose build                                       # bakes BGE-M3 + BGE-reranker-v2-m3 at pinned revisions
docker compose run --rm app tcb check                      # M0 gate: load both models, 1 embed + 1 rerank, RSS
docker compose run --rm app pytest                         # tests (index tests use a hash embedder)
mkdir -p out && chmod a+w out                              # Linux: let the container user write ./out
docker compose run --rm app tcb build data/ar.pdf out/idx  # build with BGE-M3 → ./out/idx (then: rm -rf idx && mv out/idx idx)
docker compose run --rm app tcb validate /app/idx --snapshot    # fail-closed validation of the shipped index
```
`idx/` (shipped index, ≈ 43 MB) is committed to git; the server never rebuilds it.
Without Docker: `pip install -e '.[dev]' && pytest` (needs `poppler-utils`).

## Layout
`src/tcb/`: `pdf` (layout, spreads, page map) · `parse` (KPI, FS tables, glossary) · `nodes` · `build` · `index` · `vi` (tokenizer) · `conf` (fixed models) · `models` (Docker download) · `cli`.
`data/`: `ar.pdf`, `q10.json` (published questions, never tuned on).

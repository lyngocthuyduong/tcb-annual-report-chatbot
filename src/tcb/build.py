"""Offline: PDF -> nodes -> dense (LlamaIndex) + lexical (bm25s) indexes + manifest. Run once, ship idx/."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import time
from pathlib import Path

from . import conf, nodes as N, parse, pdf, vi


class ModelMissing(RuntimeError):
    pass


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def real_embedder():
    """BGE-M3 from the baked local snapshot. Never downloads."""
    path = conf.MODELS_DIR / "bge-m3"
    if not path.exists():
        raise ModelMissing(f"{path} missing. Models are baked into the Docker image (see Dockerfile); "
                           "run inside the image: docker compose run --rm app tcb build ...")
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    return HuggingFaceEmbedding(model_name=str(path), normalize=True, embed_batch_size=8, max_length=512, device="cpu")


def _meta(n: N.Node) -> dict:
    m = {"chunk_id": n.chunk_id, "pdf_page_index": n.pdf_page_index, "printed_page_number": n.printed_page_number,
         "section": n.section, "content_type": n.content_type}
    if n.table_id: m["table_id"] = n.table_id
    return m


def build(pdf_path: str, out_dir: str, embedder=None, embed_name: str = conf.EMBED_MODEL,
          embed_rev: str = conf.EMBED_REV, force: bool = False) -> dict:
    out = Path(out_dir)
    if out.exists() and any(out.iterdir()) and not force:
        raise FileExistsError(f"{out} not empty (pass --force to overwrite)")
    out.mkdir(parents=True, exist_ok=True)
    emb = embedder or real_embedder()
    dim = len(emb.get_text_embedding("kiểm tra"))
    if embed_name == conf.EMBED_MODEL and dim != conf.EMBED_DIM:
        raise RuntimeError(f"embedding dim {dim} != {conf.EMBED_DIM}")
    t0 = time.time()

    halves, pm = pdf.load(pdf_path)
    nodes, gloss = N.build_nodes(halves)
    images = parse.image_inventory(pdf_path)
    t_ingest = time.time() - t0

    # ---- dense: LlamaIndex vector index
    from llama_index.core import StorageContext, VectorStoreIndex
    from llama_index.core.schema import TextNode
    t1 = time.time()
    tn = []
    for n in nodes:
        x = TextNode(id_=n.chunk_id, text=n.text, metadata=_meta(n))
        x.excluded_embed_metadata_keys = list(x.metadata.keys())
        x.excluded_llm_metadata_keys = list(x.metadata.keys())
        tn.append(x)
    index = VectorStoreIndex(tn, embed_model=emb, show_progress=False, storage_context=StorageContext.from_defaults())
    index.storage_context.persist(persist_dir=str(out / "llamaindex"))
    t_embed = time.time() - t1

    # ---- lexical: bm25s over our tokenizer
    import bm25s
    t2 = time.time()
    bm = bm25s.BM25(k1=conf.BM25_K1, b=conf.BM25_B)
    bm.index([vi.tokens(n.text) for n in nodes], show_progress=False)
    (out / "bm25").mkdir(exist_ok=True)
    bm.save(str(out / "bm25"))
    (out / "bm25" / "ids.json").write_text(json.dumps([n.chunk_id for n in nodes]))
    t_bm25 = time.time() - t2

    # ---- plain artifacts
    N.dump_jsonl(nodes, out / "nodes.jsonl")
    pages: dict = {}
    for h in halves:
        pages.setdefault(str(h.pdf_page_index), {})[h.side] = {"printed": h.printed_page, "source": h.page_source}
    (out / "pages.json").write_text(json.dumps(pages, ensure_ascii=False))
    (out / "gloss.json").write_text(json.dumps(gloss, ensure_ascii=False, indent=0))
    with open(out / "tables.jsonl", "w", encoding="utf-8") as f:
        for n in nodes:
            if n.content_type == "table":
                f.write(json.dumps({"table_id": n.table_id, "chunk_id": n.chunk_id, "form_code": n.form_code,
                                    "unit": n.unit_header, "printed_page": n.printed_page_number,
                                    "rows": n.rows}, ensure_ascii=False) + "\n")
    by_type: dict[str, int] = {}
    for n in nodes: by_type[n.content_type] = by_type.get(n.content_type, 0) + 1
    report = {"page_map": pm, "images": images, "nodes_by_type": by_type, "glossary_terms": len(gloss),
              "seconds": {"ingest": round(t_ingest, 1), "embed": round(t_embed, 1), "bm25": round(t_bm25, 1)},
              "hardware": {"cpu": platform.processor() or platform.machine(), "cores": os.cpu_count(),
                           "python": platform.python_version()}}
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))

    # ---- manifest (hashes every other file)
    import importlib.metadata as md
    def ver(p):
        try: return md.version(p)
        except md.PackageNotFoundError: return None
    manifest = {
        "index_format_version": conf.INDEX_FORMAT,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": {"document_id": conf.DOC_ID, "filename": Path(pdf_path).name, "sha256": sha256(Path(pdf_path)),
                   "pdf_pages": pm["pdf_pages"], "halves": pm["halves"],
                   "printed_pages_footer": pm["footer"], "printed_pages_inferred": pm["inferred"]},
        "embedding": {"model": embed_name, "revision": embed_rev, "dim": dim, "normalize": True, "max_length": 512},
        "reranker_expected": {"model": conf.RERANK_MODEL, "revision": conf.RERANK_REV},
        "chunking": {"unit": "half-page", "narrative_max_words": conf.CHUNK_MAX_WORDS,
                     "table_rows_per_chunk": conf.TABLE_ROWS_PER_CHUNK, "glossary": "one-per-term"},
        "bm25": {"library": "bm25s", "version": ver("bm25s"), "tokenizer": conf.TOKENIZER,
                 "k1": conf.BM25_K1, "b": conf.BM25_B},
        "counts": {"nodes": len(nodes), "by_type": by_type, "glossary": len(gloss)},
        "runtime_requirements": {k: ver(k) for k in ("llama-index-core", "llama-index-embeddings-huggingface",
                                                     "sentence-transformers", "bm25s", "torch", "transformers")},
        "files": {},
    }
    for p in sorted(out.rglob("*")):
        if p.is_file() and p.name != "manifest.json":
            manifest["files"][p.relative_to(out).as_posix()] = sha256(p)
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    return manifest

"""tcb CLI. M0–M2: check | build | validate. (chat/ask/batch/eval/smoke arrive in later steps.)"""
from __future__ import annotations

import argparse
import json
import resource
import sys
import time

from . import conf


def cmd_check(a) -> int:
    """M0 gate: load both fixed models from the baked snapshot, run one embed + one rerank, log time/RSS."""
    out: dict = {"embed": conf.EMBED_MODEL, "rerank": conf.RERANK_MODEL}
    for name, sub, rev in (("embed", "bge-m3", conf.EMBED_REV), ("rerank", "bge-reranker-v2-m3", conf.RERANK_REV)):
        p = conf.MODELS_DIR / sub
        if not p.exists():
            print(f"FAIL: {p} missing (models are baked into the image)", file=sys.stderr); return 3
        r = (p / ".rev")
        out[f"{name}_snapshot_rev_ok"] = r.exists() and r.read_text().strip() == rev
    t = time.time()
    from sentence_transformers import CrossEncoder, SentenceTransformer
    m = SentenceTransformer(str(conf.MODELS_DIR / "bge-m3"), device="cpu")
    out["embed_load_s"] = round(time.time() - t, 1)
    t = time.time()
    v = m.encode(["Tỷ lệ CASA của Techcombank năm 2025 là bao nhiêu?", "Tỷ lệ CASA 40,4% dẫn đầu ngành."],
                 normalize_embeddings=True)
    out["embed_dim"] = int(v.shape[1]); out["embed_s"] = round(time.time() - t, 2)
    out["cosine_pos"] = float(v[0] @ v[1])
    t = time.time()
    ce = CrossEncoder(str(conf.MODELS_DIR / "bge-reranker-v2-m3"), device="cpu", max_length=512)
    out["rerank_load_s"] = round(time.time() - t, 1)
    t = time.time()
    s = ce.predict([("Tỷ lệ CASA năm 2025?", "Tỷ lệ CASA 40,4% dẫn đầu ngành."),
                    ("Tỷ lệ CASA năm 2025?", "Giải thưởng Nơi làm việc tốt nhất Đông Nam Á.")])
    out["rerank_s"] = round(time.time() - t, 2); out["rerank_scores"] = [float(x) for x in s]
    out["rerank_pos_gt_neg"] = bool(s[0] > s[1])
    out["peak_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    print(json.dumps(out, indent=1, ensure_ascii=False))
    return 0 if out["embed_dim"] == conf.EMBED_DIM and out["rerank_pos_gt_neg"] and out["embed_snapshot_rev_ok"] \
        and out["rerank_snapshot_rev_ok"] else 1


def cmd_build(a) -> int:
    from .build import build
    m = build(a.pdf, a.out, force=a.force)
    print(json.dumps({"nodes": m["counts"], "files": len(m["files"])}, ensure_ascii=False))
    return 0


def cmd_validate(a) -> int:
    from .index import validate
    m = validate(a.dir, check_snapshot=a.snapshot)
    print(f"OK nodes={m['counts']['nodes']} embed={m['embedding']['model']}@{m['embedding']['revision'][:7]}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="tcb")
    sp = p.add_subparsers(dest="cmd", required=True)
    sp.add_parser("check").set_defaults(f=cmd_check)
    b = sp.add_parser("build"); b.add_argument("pdf"); b.add_argument("out"); b.add_argument("--force", action="store_true")
    b.set_defaults(f=cmd_build)
    v = sp.add_parser("validate"); v.add_argument("dir", nargs="?", default=str(conf.settings().index_dir))
    v.add_argument("--snapshot", action="store_true", help="also check baked model snapshot revision")
    v.set_defaults(f=cmd_validate)
    a = p.parse_args(argv)
    from .index import IndexError_
    try:
        return a.f(a)
    except IndexError_ as e:
        print(f"INDEX ERROR {e}", file=sys.stderr); return 3


if __name__ == "__main__":
    raise SystemExit(main())

"""tcb CLI: doctor | models | check | build | validate | ask | chat | batch | eval | exp | vset | smoke | serve."""
from __future__ import annotations

import argparse
import json
import sys
import time

from . import conf


def peak_rss_mb() -> int | None:
    """Peak resident memory of this process; None where the OS gives no cheap answer (Windows without psutil)."""
    try:
        import resource                                   # Unix only
        kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return round(kb / (1024 * 1024) if sys.platform == "darwin" else kb / 1024)   # macOS reports bytes, Linux KiB
    except ImportError:
        try:
            import psutil
            return round(psutil.Process().memory_info().peak_wset / 1e6)
        except Exception:
            return None


def cmd_check(a) -> int:
    """M0 gate: load the fixed embedding model from the baked snapshot, embed two texts and one query, log time/RSS."""
    out: dict = {"embed": conf.EMBED_MODEL}
    p = conf.MODELS_DIR / "bge-m3"
    if not p.exists():
        print(f"FAIL: {p} missing (models are baked into the image)", file=sys.stderr); return 3
    r = p / ".rev"
    out["embed_snapshot_rev_ok"] = r.exists() and r.read_text(encoding="utf-8").strip() == conf.EMBED_REV
    t = time.time()
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(str(p), device="cpu")
    out["embed_load_s"] = round(time.time() - t, 1)
    v = m.encode(["Tỷ lệ CASA của Techcombank năm 2025 là bao nhiêu?", "Tỷ lệ CASA 40,4% dẫn đầu ngành.",
                  "Giải thưởng Nơi làm việc tốt nhất Đông Nam Á."], normalize_embeddings=True)
    out["embed_dim"] = int(v.shape[1])
    out["cosine_pos"], out["cosine_neg"] = float(v[0] @ v[1]), float(v[0] @ v[2])
    t = time.time(); m.encode(["Tổng tài sản của Techcombank tại ngày 31/12/2025 là bao nhiêu?"]); out["query_embed_s"] = round(time.time() - t, 3)
    out["peak_rss_mb"] = peak_rss_mb()
    print(json.dumps(out, indent=1, ensure_ascii=False))
    return 0 if out["embed_dim"] == conf.EMBED_DIM and out["cosine_pos"] > out["cosine_neg"] and out["embed_snapshot_rev_ok"] else 1


def cmd_vset(a) -> int:
    from . import nodes as nodes_, pdf, vset
    def parsed():
        halves, _ = pdf.load(a.pdf)
        return nodes_.build_nodes(halves)[0]
    bad = vset.check(a.file, parsed())
    print("\n".join(bad) if bad else f"OK: gold facts of {a.file} are on their gold pages")
    return 1 if bad else 0


def cmd_build(a) -> int:
    from .build import build
    m = build(a.pdf, a.out, force=a.force, chunk_words=a.chunk_words)
    print(json.dumps({"nodes": m["counts"], "files": len(m["files"])}, ensure_ascii=False))
    return 0


def cmd_validate(a) -> int:
    from .index import validate
    m = validate(a.dir, check_snapshot=a.snapshot)
    print(f"OK nodes={m['counts']['nodes']} embed={m['embedding']['model']}@{m['embedding']['revision'][:7]}")
    return 0


def _engine():
    from .boot import load_engine
    e = load_engine()
    e.rt.warm()
    return e


def cmd_ask(a) -> int:
    ans = _engine().ask(a.question)
    print(json.dumps(ans.to_dict(trace=a.trace), ensure_ascii=False, indent=1) if a.json or a.trace else
          f"{ans.answer}\n\n[{ans.status} | {ans.provider}/{ans.model}]")
    return 0 if ans.status != "error" else 4


def cmd_chat(a) -> int:
    from . import chat
    e, s = _engine(), chat.Session()
    print(f"{e.prov.name}/{e.prov.model} — gõ câu hỏi (Ctrl-D để thoát)")
    while True:
        try: q = input("> ").strip()
        except EOFError: return 0
        if not q: continue
        ans = e.ask(q, s.turns)
        print(f"{ans.answer}\n[{ans.status}]")
        if ans.status != "error": s.add(chat.Turn(q, ans.standalone, ans.answer, ans.status))


def cmd_batch(a) -> int:
    from pathlib import Path
    qs = json.loads(Path(a.file).read_text(encoding="utf-8"))
    e, out = _engine(), Path(a.out); out.mkdir(parents=True, exist_ok=True)
    n_err = 0
    with open(out / "answers.jsonl", "w", encoding="utf-8") as f:
        for i, q in enumerate(qs, 1):
            ans = e.ask(q["question"])
            n_err += ans.status == "error"
            f.write(json.dumps({"id": q.get("id", f"q{i}"), **ans.to_dict(trace=True)}, ensure_ascii=False) + "\n")
            print(f"{i}/{len(qs)} {q.get('id', '')} {ans.status}", flush=True)
    return 4 if n_err else 0


def cmd_eval(a) -> int:
    from . import ev
    if a.compare:                                           # no engine, no models: just diff two finished runs
        c = ev.compare(*a.compare)
        print(ev.compare_text(c))
        from pathlib import Path
        out = Path(a.out) / f"compare_{Path(a.compare[0]).name}_vs_{Path(a.compare[1]).name}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(c, ensure_ascii=False, indent=1), encoding="utf-8")
        return 1 if c["regressed"] else 0
    if not a.set:
        print("eval needs a dataset (or --compare RUN_A RUN_B)", file=sys.stderr); return 2
    rep = ev.run_eval(a.set, a.out, _engine(), label=a.label, include_unverified=a.include_unverified)
    print(json.dumps(rep["summary"], ensure_ascii=False, indent=1))
    return 0


def cmd_exp(a) -> int:
    from . import boot, exp
    rt = boot.load_retriever(a.index, lenient=True)
    ctx = exp.make_ctx(rt.ix, rt.gloss)
    which = tuple(x.strip() for x in a.exp.split(",") if x.strip())
    bad = [x for x in which if x not in exp.EXPS]
    if bad:
        print(f"unknown experiment(s) {bad}; choose from {exp.EXPS}", file=sys.stderr); return 2
    res = exp.run_exp(a.sets, ctx, which, a.include_unverified, progress=lambda s: print(s, flush=True) if a.verbose else None)
    res["meta"]["index"] = str(a.index)
    d = exp.write(res, a.out, a.label)
    print(json.dumps(res["decisions"], ensure_ascii=False, indent=1)); print(f"written: {d}/exp.json  {d}/exp.md")
    return 0


def cmd_smoke(a) -> int:
    from . import smoke
    return smoke.run(a.out)


def cmd_doctor(a) -> int:
    from . import doctor
    return doctor.run(a.json)


def cmd_models(a) -> int:
    from . import models
    {"download": models.download, "convert": models.convert}[a.action]()
    return 0


def cmd_serve(a) -> int:
    import uvicorn
    from .api import create_app
    st = conf.settings()
    app = create_app(_engine(), st.trace_dir)          # config/index errors surface here, before the port opens
    uvicorn.run(app, host=st.host, port=st.port, log_level=st.log_level.lower())
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="tcb")
    sp = p.add_subparsers(dest="cmd", required=True)
    sp.add_parser("check").set_defaults(f=cmd_check)
    b = sp.add_parser("build"); b.add_argument("pdf"); b.add_argument("out"); b.add_argument("--force", action="store_true")
    b.add_argument("--chunk-words", type=int, default=None, help="EXPERIMENT E6 only: narrative chunk size; such an index is rejected by `serve`/`validate`")
    b.set_defaults(f=cmd_build)
    v = sp.add_parser("validate"); v.add_argument("dir", nargs="?", default=str(conf.settings().index_dir))
    v.add_argument("--snapshot", action="store_true", help="also check baked model snapshot revision")
    v.set_defaults(f=cmd_validate)
    k = sp.add_parser("ask"); k.add_argument("question"); k.add_argument("--json", action="store_true")
    k.add_argument("--trace", action="store_true"); k.set_defaults(f=cmd_ask)
    sp.add_parser("chat").set_defaults(f=cmd_chat)
    bt = sp.add_parser("batch"); bt.add_argument("file"); bt.add_argument("--out", default="out/runs/batch"); bt.set_defaults(f=cmd_batch)
    ev_ = sp.add_parser("eval"); ev_.add_argument("set", nargs="?"); ev_.add_argument("--out", default="out/runs/eval")
    ev_.add_argument("--compare", nargs=2, metavar=("RUN_A", "RUN_B"), help="diff two finished run folders (A=baseline)")
    ev_.add_argument("--include-unverified", action="store_true")
    ev_.add_argument("--label", default=""); ev_.set_defaults(f=cmd_eval)
    ex = sp.add_parser("exp", help="retrieval experiments (E2,E3,E3b,E4,E5); no LLM, no token cost")
    ex.add_argument("sets", nargs="+", help="verified dev/val JSON files, e.g. data/dev.json data/val.json")
    ex.add_argument("--exp", default=",".join(__import__("tcb.exp", fromlist=["EXPS"]).EXPS)); ex.add_argument("--index", default=str(conf.settings().index_dir))
    ex.add_argument("--out", default="out/exp"); ex.add_argument("--label", default="m3"); ex.add_argument("--include-unverified", action="store_true")
    ex.add_argument("--verbose", action="store_true")
    ex.set_defaults(f=cmd_exp)
    vs = sp.add_parser("vset", help="dev/val check: gold facts (or calc inputs) are on the gold pages of the parsed PDF")
    vs.add_argument("action", choices=["check"]); vs.add_argument("file", nargs="?", default="data/dev.json")
    vs.add_argument("--pdf", default="data/ar.pdf"); vs.set_defaults(f=cmd_vset)
    sm = sp.add_parser("smoke", help="Ollama smoke test (needs a running Ollama with the model pulled)")
    sm.add_argument("--out", default="out/smoke.json"); sm.set_defaults(f=cmd_smoke)
    sp.add_parser("serve").set_defaults(f=cmd_serve)
    dr = sp.add_parser("doctor", help="check this machine: python, packages, poppler, PDF, models, index, keys, RAM/disk, Docker")
    dr.add_argument("--json", action="store_true"); dr.set_defaults(f=cmd_doctor)
    mo = sp.add_parser("models", help="download/convert the two fixed models into TCB_MODELS_DIR (needs network)")
    mo.add_argument("action", choices=["download", "convert"]); mo.set_defaults(f=cmd_models)
    a = p.parse_args(argv)
    conf.load_env()                                         # local runs: pick up ./.env (Docker uses env_file)
    from .index import IndexError_
    from .llm import ConfigError
    try:
        return a.f(a)
    except IndexError_ as e:
        print(f"INDEX ERROR {e}", file=sys.stderr); return 3
    except ConfigError as e:
        print(f"CONFIG ERROR {e}", file=sys.stderr); return 2
    except Exception as e:                                  # model snapshot missing etc.: clear message, no traceback wall
        from .build import ModelMissing
        if isinstance(e, ModelMissing):
            print(f"MODEL ERROR {e}", file=sys.stderr); return 5
        raise


if __name__ == "__main__":
    raise SystemExit(main())

"""Evaluation: dataset schema, per-item scoring, stage attribution, machine-readable traces, summary.
Deterministic scorers only (no LLM judge). Items count toward reported metrics only when `verified` is true
(a human checked the gold against the PDF); unverified items are run and listed separately."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import statistics
import subprocess
import time
from pathlib import Path
from typing import Callable, Literal

from pydantic import BaseModel, Field

from . import chat, cite, conf, vi


class Turn_(BaseModel):
    question: str
    answer: str


class Item(BaseModel):
    id: str
    category: str = ""
    type: Literal["single", "multi", "calc", "followup", "refusal"] = "single"
    question: str
    gold_answer: str = ""
    gold_printed_pages: list[int] = Field(default_factory=list)
    alt_pages: list[int] = Field(default_factory=list)       # other printed pages that also legitimately answer
    answerable: bool = True
    gold_facts: list[str] = Field(default_factory=list)      # each must appear in the answer; "a | b" = either
    history: list[Turn_] = Field(default_factory=list)       # prior turns for follow-up items
    verified: bool = False                                    # checked against the PDF text (notes say by whom)
    expect_status: Literal["", "ambiguous"] = ""              # "ambiguous": the right behaviour is a clarifying question
    notes: str = ""


def load_items(path: str | Path) -> list[Item]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    items = [Item.model_validate(x) for x in raw]
    ids = [i.id for i in items]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate item ids")
    for i in items:
        if not i.answerable and i.gold_printed_pages:
            raise ValueError(f"{i.id}: unanswerable item must not have gold pages")
    return items


# ------------------------------------------------------------------ scorers
def _fold(s: str) -> str:
    return " ".join(vi.fold(vi.nfc(s).lower()).split())


def _scales(text: str) -> dict[str, set[str]]:
    """canonical number -> magnitudes printed right after it in `text` ('nghìn tỷ', 'triệu', '%' ...)."""
    out: dict[str, set[str]] = {}
    t = vi.nfc(text)
    for m in cite._NUM.finditer(t):
        s = cite._scale_at(t, m.start(), m.end())
        out.setdefault(vi.canon_num(m.group(0)).rstrip("%"), set()).update({s} if s else set())
    return out


def facts_ok(item: Item, answer: str) -> tuple[bool | None, str]:
    """-> (ok | None if not machine-scorable, method)."""
    facts = item.gold_facts
    method = "gold_facts"
    if not facts:
        method = "gold_answer_numbers"
        gn = {t for t in cite.num_tokens(item.gold_answer) if not cite._trivial(t)}
        if not gn:
            return None, "unscorable"
        an = cite.num_tokens(answer)
        # a gold number is satisfied by its raw or canonical form
        ok = all(t in an or vi.canon_num(t).rstrip("%") in an for t in {vi.canon_num(x).rstrip("%") for x in gn})
        return ok, method
    an, af = cite.num_tokens(answer), _fold(answer)
    a_sc, g_sc = _scales(answer), _scales(item.gold_answer)
    a_fig = cite.figures(answer, small=True)

    def same_value(t: str) -> bool:
        """The gold figure stated more precisely or in another unit: '53,4 nghìn tỷ' ≈ '53.391 tỷ' (within half a unit of
        the gold's last digit). An answer less precise than the gold ('40%' for '40,4%') does not count. A gold count
        without a magnitude that ends in 3+ zeros is read as rounded to the thousand ('hơn 303.000' is met by '303.446');
        a word scale is an exact conversion ('1 triệu' = '1.000.000', never '600.000'); otherwise it only matches exactly."""
        g = []
        for f in cite.figures(item.gold_answer, keep=t):
            if f[2] == cite.Decimal("0.5"):
                z = len(t) - len(t.rstrip("0")) if t.isdigit() else 0
                if z < 3: continue
                f = (f[0], f[1], cite.Decimal(500))
            g.append(f)
        return any(f[0] == x[0] and abs(f[1] - x[1]) <= f[2] for f in g for x in a_fig)

    def has(f: str) -> bool:
        fn = {t for t in cite.num_tokens(f) if not cite._trivial(t)}
        if fn:   # the digits with the gold's magnitude (100 tỷ is not 100 triệu), or the same value at another rounding
            return all((t in an and (not g_sc.get(t) or g_sc[t] & a_sc.get(t, set()))) or same_value(t) for t in fn)
        return _fold(f) in af
    return all(any(has(alt.strip()) for alt in f.split(" | ")) for f in facts), method


def cited_pages(answer: str) -> list[int]:
    return sorted({int(p) for m in cite.CITE_RE.finditer(answer) for p in m.group(1).split(", ")})


def stage_hits(gold: set[int], tr: dict, page_of: Callable[[str], int | None]) -> dict:
    """First rank (1-based) at which a gold page appears, per stage; None = absent."""
    r = tr.get("retrieval", {})
    def first(ids):
        for i, c in enumerate(ids, 1):
            if page_of(c) in gold: return i
        return None
    return {"bm25": first(r.get("bm25", [])), "dense": first(r.get("dense", [])), "fused": first(r.get("fused", [])),
            "final": first([c for c, _ in r.get("ranked", [])])}


def attribute(item: Item, ans: dict, hits: dict | None, fact: bool | None, cite_bad: list[int]) -> str:
    """First failing stage for a failed item; 'pass' otherwise. Order: system -> retrieval -> validation -> generation -> refusal."""
    st = ans["status"]
    if st == "error":
        return f"system:{ans.get('error_kind') or 'error'}"
    if item.expect_status:
        return "pass" if st == item.expect_status else f"refusal:expected_{item.expect_status}"
    if not item.answerable:
        return "pass" if st == "not_in_report" else "refusal:under_refusal"
    if cite_bad:
        return "citation:invalid_page"
    if hits is not None:
        union = [hits["bm25"], hits["dense"]]
        if all(x is None for x in union): return "retrieval:recall_miss"
        if hits["fused"] is None: return "retrieval:fusion_drop"
        if hits["final"] is None or hits["final"] > conf.FINAL_EVIDENCE: return "retrieval:rank_drop"
    if st == "not_in_report":
        return "refusal:over_refusal" if hits is not None else "refusal:over_refusal_no_gold"
    if ans.get("trace", {}).get("calc_errors"): return "calc:error"
    if fact is False:
        return "generation:wrong_or_missing_fact"
    cited = set(cited_pages(ans["answer"]))
    if item.gold_printed_pages and not (cited & (set(item.gold_printed_pages) | set(item.alt_pages))):
        return "citation:wrong_page"
    return "pass"


def score_item(item: Item, ans: dict, page_of: Callable[[str], int | None], declared: set[int] | None = None) -> dict:
    """declared = the gold and alt pages written in the set, before `widen_gold`; page_hit_declared counts only those."""
    tr = ans.get("trace", {})
    gold = set(item.gold_printed_pages) | set(item.alt_pages)
    hits = stage_hits(gold, tr, page_of) if (item.answerable and gold) else None
    fact, method = facts_ok(item, ans["answer"]) if item.answerable else (None, "n/a")
    ev_pages = {e["page"] for e in tr.get("evidence", []) if e.get("page") is not None}
    cited = cited_pages(ans["answer"])
    bad = [p for p in cited if p not in ev_pages]
    verdict = attribute(item, ans, hits, fact, bad)
    rr = None
    if hits and hits["final"]: rr = 1 / hits["final"]
    llm_ = tr.get("llm", {})
    return {"id": item.id, "category": item.category, "type": item.type, "answerable": item.answerable,
            "expect_status": item.expect_status, "verified": item.verified, "status": ans["status"], "verdict": verdict, "passed": verdict == "pass",
            "fact_ok": fact, "fact_method": method, "cited_pages": cited, "gold_pages": sorted(gold),
            "citation_valid": not bad, "page_hit": bool(set(cited) & gold) if gold else None,
            "page_hit_declared": bool(set(cited) & (declared if declared is not None else gold)) if gold else None,
            "page_precision": (len(set(cited) & gold) / len(cited)) if (gold and cited) else None,
            "stage_ranks": hits, "rr": rr, "rewrite_mode": tr.get("rewrite_mode"),
            "latency_s": tr.get("total_s"), "llm_calls": llm_.get("calls"), "cost_usd": llm_.get("cost_usd"),
            "in_tokens": llm_.get("in_tokens"), "out_tokens": llm_.get("out_tokens"), "repaired": tr.get("repaired"),
            "stage_s": {**tr.get("retrieval", {}).get("seconds", {}), "llm": llm_.get("latency_s")},
            "why": {"plan_status": (tr.get("plan") or {}).get("status"), "decision": (tr.get("decision") or {}).get("reason"),
                    "problems": (tr.get("check_problems") or [])[:3], "calc_errors": tr.get("calc_errors") or [],
                    "error": ans.get("error")}}


def _rate(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(map(bool, xs)) / len(xs), 4) if xs else None


def _pct(xs, p):
    xs = sorted(x for x in xs if x is not None)
    return round(xs[min(len(xs) - 1, int(p * len(xs)))], 3) if xs else None


def failure_lines(rows: list[dict]) -> list[str]:
    """One line per failed item: what the eval saw and why the pipeline decided as it did (from the trace)."""
    out = []
    for r in rows:
        if r["passed"]: continue
        w = r.get("why", {})
        out.append(f"{r['id']}: {r['verdict']} | status={r['status']} plan={w.get('plan_status')} decision={w.get('decision')} "
                   f"| gold rank {(r.get('stage_ranks') or {}).get('final')} | problems={w.get('problems')} calc={w.get('calc_errors')}"
                   + (f" | error={w['error']}" if w.get("error") else ""))
    return out


def summarize(rows: list[dict]) -> dict:
    ans = [r for r in rows if r["answerable"]]
    un = [r for r in rows if not r["answerable"] and not r.get("expect_status")]
    gold = [r for r in ans if r["stage_ranks"]]
    refused = [r for r in rows if r["status"] == "not_in_report"]
    def recall(stage, k):
        return _rate([(r["stage_ranks"][stage] or 10**9) <= k for r in gold])
    s = {
        "n": len(rows), "pass_rate": _rate([r["passed"] for r in rows]),
        "retrieval": {"n_with_gold_pages": len(gold), "recall@40_bm25": recall("bm25", 40), "recall@40_dense": recall("dense", 40),
                      f"recall@{conf.TOP_BM25}_fused": recall("fused", conf.TOP_BM25),
                      f"recall@{conf.FINAL_EVIDENCE}_final": recall("final", conf.FINAL_EVIDENCE),
                      "recall@3_final": recall("final", 3),
                      "mrr_final": round(statistics.mean([r["rr"] or 0 for r in gold]), 4) if gold else None},
        "answers": {"fact_correct": _rate([r["fact_ok"] for r in ans]),
                    "fact_unscorable": sum(r["fact_ok"] is None for r in ans),
                    "citation_valid": _rate([r["citation_valid"] for r in rows if r["status"] != "error"]),
                    "page_hit": _rate([r["page_hit"] for r in ans]),
                    "page_hit_declared": _rate([r.get("page_hit_declared") for r in ans]),
                    "page_precision": round(statistics.mean([r["page_precision"] for r in ans if r["page_precision"] is not None]), 4)
                    if any(r["page_precision"] is not None for r in ans) else None},
        "refusal": {"precision": _rate([not r["answerable"] and not r.get("expect_status") for r in refused]),
                    "recall": _rate([r["status"] == "not_in_report" for r in un]),
                    "n_unanswerable": len(un), "false_answer_rate": _rate([r["status"] in ("answered", "partial") for r in un])},
        "latency_s": {"mean": round(statistics.mean(xs), 3) if (xs := [r["latency_s"] for r in rows if r["latency_s"] is not None]) else None,
                      "p50": _pct([r["latency_s"] for r in rows], .5), "p95": _pct([r["latency_s"] for r in rows], .95)},
        # cost is None (not 0) when the model has no price in conf.PRICES; tokens are always reported
        "tokens_in_total": sum(r["in_tokens"] or 0 for r in rows), "tokens_out_total": sum(r["out_tokens"] or 0 for r in rows),
        "cost_usd_total": None if any(r["llm_calls"] and r["cost_usd"] is None for r in rows) else round(sum(r["cost_usd"] or 0 for r in rows), 5),
        "cost_usd_per_question": None if any(r["llm_calls"] and r["cost_usd"] is None for r in rows) or not rows
        else round(sum(r["cost_usd"] or 0 for r in rows) / len(rows), 6),
        "stage_seconds_p50": {k: _pct([r["stage_s"].get(k) for r in rows], .5) for k in ("embed", "dense", "bm25", "llm")},
        "llm_calls_mean": round(statistics.mean([r["llm_calls"] or 0 for r in rows]), 2) if rows else None,
        "attribution": {},
    }
    for r in rows:
        s["attribution"][r["verdict"]] = s["attribution"].get(r["verdict"], 0) + 1
    for key in ("category", "type"):
        g: dict = {}
        for r in rows: g.setdefault(r[key] or "-", []).append(r["passed"])
        s["by_" + key] = {k: {"n": len(v), "pass_rate": _rate(v)} for k, v in sorted(g.items())}
    return s


# ------------------------------------------------------------------ runner
def widen_gold(item: Item, nodes: dict) -> Item:
    """The report repeats many figures (p.5 KPI tile, p.52 review, p.297 statement). A citation is correct if the analyst
    can read the figure on the cited page, so every page with one node holding ALL numeric gold facts counts as gold too.
    Only for items whose facts are all numbers (a word-only fact could match unrelated text) and include at least one
    distinctive number (a decimal or 4+ digits)."""
    if not item.answerable or not item.gold_facts:
        return item
    alts = [[{t for t in cite.num_tokens(a) if not cite._trivial(t)} for a in f.split(" | ")] for f in item.gold_facts]
    if any(not all(alt) for alt in alts):
        return item
    distinctive = lambda t: "." in t or len(t) >= 4                     # '21.8', '1192'; not '100' or '302'
    if not any(distinctive(t) for alt in alts for a in alt for t in a):
        return item                                                   # a short integer recurs on unrelated pages
    pages = set()
    for n in nodes.values():
        toks = cite.num_tokens(n["text"])
        if all(any(a <= toks for a in alt) for alt in alts):
            pages.add(n["printed_page_number"])
    extra = sorted(pages - set(item.gold_printed_pages) - set(item.alt_pages))
    return item.model_copy(update={"alt_pages": item.alt_pages + extra}) if extra else item


def _commit() -> str | None:
    """The code that produced the numbers (None outside a git checkout, e.g. inside the Docker image)."""
    try:
        return subprocess.run(["git", "describe", "--always", "--dirty"], capture_output=True, text=True, timeout=5).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def run_eval(set_path: str, out_dir: str, engine, label: str = "", include_unverified: bool = False) -> dict:
    """Run every item through `engine`, write traces/results/summary/config, return the report."""
    nodes = engine.rt.ix.nodes
    raw = load_items(set_path)
    declared = {it.id: set(it.gold_printed_pages) | set(it.alt_pages) for it in raw}
    items = [widen_gold(it, nodes) for it in raw]
    page_of = lambda c: (nodes.get(c) or {}).get("printed_page_number")
    out = Path(out_dir) / (label or time.strftime("%Y%m%d-%H%M%S"))
    out.mkdir(parents=True, exist_ok=True)
    rows, t0 = [], time.time()
    with open(out / "traces.jsonl", "w", encoding="utf-8") as tf:
        for it in items:
            turns = [chat.Turn(h.question, h.question, h.answer, "answered") for h in it.history]
            a = engine.ask(it.question, turns).to_dict(trace=True)
            row = score_item(it, a, page_of, declared[it.id])
            rows.append(row)
            tf.write(json.dumps({"id": it.id, "question": it.question, "gold": it.model_dump(), "answer": a, "score": row},
                                ensure_ascii=False) + "\n")
    counted = [r for r in rows if r["verified"] or include_unverified]
    m = engine.rt.ix.manifest
    cfg = {"label": label, "set": str(set_path), "set_sha256": hashlib.sha256(Path(set_path).read_bytes()).hexdigest(),
           "provider": engine.prov.name, "model": engine.prov.model,
           "index_embedding": m["embedding"], "index_built_at": m["built_at"], "index_nodes": m["counts"]["nodes"],
           "final_evidence": engine.rt.k, "device": os.getenv("TCB_DEVICE", "cpu"), "rrf_k": engine.rt.rrf_k,
           "top_bm25": engine.rt.top_bm25, "top_dense": engine.rt.top_dense, "include_unverified": include_unverified,
           "submission_grade": (m["embedding"]["model"], m["embedding"].get("revision")) == (conf.EMBED_MODEL, conf.EMBED_REV),
           "commit": _commit(),                                        # '-dirty' when the working tree had uncommitted changes
           "index_manifest_sha256": hashlib.sha256(json.dumps(m, sort_keys=True).encode()).hexdigest(),
           "hardware": {"cpu": platform.machine(), "cores": os.cpu_count()},
           "seconds": round(time.time() - t0, 1)}
    summary = summarize(counted) if counted else {"n": 0, "note": "no verified items; use include_unverified for a pipeline check"}
    summary["items_run"] = len(rows)
    summary["items_unverified_excluded"] = len(rows) - len(counted)
    summary["submission_grade"] = cfg["submission_grade"]
    (out / "results.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    (out / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    return {"summary": summary, "config": cfg, "rows": rows, "dir": str(out)}


def rescore(run_dir: str, out_dir: str, label: str, nodes: dict) -> dict:
    """Score a finished run again with the current scorer: the saved answers and frozen gold items are reused, no model is
    called. Used when the scorer changes, so the change is visible run by run (`rescored_from` in config.json)."""
    src = Path(run_dir)
    page_of = lambda c: (nodes.get(c) or {}).get("printed_page_number")
    out = Path(out_dir) / label
    out.mkdir(parents=True, exist_ok=True)
    rows, lines = [], []
    cfg0 = json.loads((src / "config.json").read_text(encoding="utf-8"))
    sp = Path(cfg0.get("set", ""))
    declared = {it.id: set(it.gold_printed_pages) | set(it.alt_pages) for it in load_items(str(sp))} if sp.is_file() else {}
    for ln in (src / "traces.jsonl").read_text(encoding="utf-8").splitlines():
        if not ln.strip(): continue
        t = json.loads(ln)
        row = score_item(Item(**t["gold"]), t["answer"], page_of, declared.get(t["id"]))
        rows.append(row); lines.append(json.dumps({**t, "score": row}, ensure_ascii=False))
    cfg = json.loads((src / "config.json").read_text(encoding="utf-8"))
    cfg.update(label=label, rescored_from=src.name, scorer_commit=_commit())
    counted = [r for r in rows if r["verified"] or cfg.get("include_unverified")]
    summary = summarize(counted)
    summary.update(items_run=len(rows), items_unverified_excluded=len(rows) - len(counted), submission_grade=cfg.get("submission_grade"))
    (out / "traces.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "results.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    (out / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    return {"summary": summary, "config": cfg, "rows": rows, "dir": str(out)}


# ------------------------------------------------------------------ run-to-run comparison
def _load_run(d: str | Path) -> tuple[dict, dict, dict]:
    d = Path(d)
    rows = {}
    for ln in (d / "results.jsonl").read_text(encoding="utf-8").splitlines():
        if ln.strip():
            r = json.loads(ln); rows[r["id"]] = r
    return rows, json.loads((d / "summary.json").read_text(encoding="utf-8")), json.loads((d / "config.json").read_text(encoding="utf-8"))


def compare(a_dir: str, b_dir: str) -> dict:
    """A = baseline, B = candidate. Reports items fixed / regressed / verdict-changed and metric deltas."""
    ra, sa, ca = _load_run(a_dir)
    rb, sb, cb = _load_run(b_dir)
    common = sorted(set(ra) & set(rb))
    fixed = [i for i in common if not ra[i]["passed"] and rb[i]["passed"]]
    regressed = [i for i in common if ra[i]["passed"] and not rb[i]["passed"]]
    changed = [{"id": i, "from": ra[i]["verdict"], "to": rb[i]["verdict"]} for i in common
               if ra[i]["verdict"] != rb[i]["verdict"] and (ra[i]["passed"] == rb[i]["passed"])]
    def delta(path):
        def get(s):
            for k in path: s = s.get(k) if isinstance(s, dict) else None
            return s
        x, y = get(sa), get(sb)
        return {"a": x, "b": y, "delta": round(y - x, 4) if isinstance(x, (int, float)) and isinstance(y, (int, float)) else None}
    metrics = {"pass_rate": ["pass_rate"], f"recall@{conf.FINAL_EVIDENCE}_final": ["retrieval", f"recall@{conf.FINAL_EVIDENCE}_final"], "mrr_final": ["retrieval", "mrr_final"],
               "fact_correct": ["answers", "fact_correct"], "citation_valid": ["answers", "citation_valid"],
               "page_hit": ["answers", "page_hit"], "refusal_precision": ["refusal", "precision"], "refusal_recall": ["refusal", "recall"],
               "false_answer_rate": ["refusal", "false_answer_rate"], "latency_p50_s": ["latency_s", "p50"],
               "latency_p95_s": ["latency_s", "p95"], "cost_usd_per_question": ["cost_usd_per_question"]}
    warn = []
    for k in ("set_sha256", "provider", "model", "final_evidence", "rrf_k"):
        if ca.get(k) != cb.get(k): warn.append(f"{k}: {ca.get(k)} -> {cb.get(k)}")
    if ca.get("index_embedding") != cb.get("index_embedding") or ca.get("index_built_at") != cb.get("index_built_at"):
        warn.append("index differs")
    if ca.get("set_sha256") != cb.get("set_sha256"):
        warn.append("DIFFERENT DATASET: deltas below are not like-for-like")
    return {"a": str(a_dir), "b": str(b_dir), "n_common": len(common), "only_in_a": sorted(set(ra) - set(rb)),
            "only_in_b": sorted(set(rb) - set(ra)), "fixed": fixed, "regressed": regressed, "verdict_changed": changed,
            "metrics": {k: delta(v) for k, v in metrics.items()}, "config_differences": warn}


def compare_text(c: dict) -> str:
    L = [f"A={c['a']}  B={c['b']}  common={c['n_common']}"]
    L += [f"! {w}" for w in c["config_differences"]]
    L.append(f"fixed ({len(c['fixed'])}): {', '.join(c['fixed']) or '-'}")
    L.append(f"REGRESSED ({len(c['regressed'])}): {', '.join(c['regressed']) or '-'}")
    for x in c["verdict_changed"]: L.append(f"  {x['id']}: {x['from']} -> {x['to']}")
    L.append("metric                      A        B        delta")
    for k, m in c["metrics"].items(): L.append(f"{k:<26} {str(m['a']):<8} {str(m['b']):<8} {m['delta']}")
    return "\n".join(L)

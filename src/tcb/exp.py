"""Retrieval experiments: E2 tokenizer, E3 fusion, E3b RRF k, E4 final k, E5 glossary expansion. No cross-encoder (DESIGN 17.1).
Retrieval only: no LLM, no tokens spent. Gold = printed pages (gold + alt) of verified answerable items.
The script SUGGESTS decisions by the plan's rules; it never edits conf.py. Tokenizer/chunk winners need a rebuild."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from . import conf, ev, find, vi

KS = (5, 8, 20, 40)
EXPS = ("E2", "E3", "E3b", "E4", "E5", "E8")
RRF_KS = (10, 30, 60, 100)
FINAL_KS = (5, 8, 10, 12, 15, 20)


# ------------------------------------------------------------------ tokenizer variants (only "vi" is production)
def tok_vi(t: str) -> list[str]:
    return vi.tokens(t)


def tok_bigram(t: str) -> list[str]:
    """vi tokens + bigrams of adjacent ORIGINAL word syllables (not the folded duplicates vi.tokens appends)."""
    raw = [m.group(0) for m in vi._TOK.finditer(vi.nfc(t).lower())]
    return vi.tokens(t) + [f"{a}_{b}" for a, b in zip(raw, raw[1:]) if a.isalpha() and b.isalpha()]


def tok_default(t: str) -> list[str]:
    import bm25s
    return bm25s.tokenize([t], return_ids=False, show_progress=False)[0]


TOKENIZERS: dict[str, Callable[[str], list[str]]] = {"default": tok_default, "vi": tok_vi, "vi+bigram": tok_bigram}


# ------------------------------------------------------------------ fusion
def fuse_rrf(lists: list[list[tuple[str, float]]], k: int) -> list[str]:
    return [c for c, _ in find.rrf([[c for c, _ in l] for l in lists], k)]


def fuse_rsf(lists: list[list[tuple[str, float]]]) -> list[str]:
    """Relative-score fusion: min-max normalise each list, sum, sort."""
    tot: dict[str, float] = {}
    for l in lists:
        if not l: continue
        s = [x for _, x in l]; lo, hi = min(s), max(s)
        for c, x in l:
            tot[c] = tot.get(c, 0.0) + ((x - lo) / (hi - lo) if hi > lo else 1.0)
    return [c for c, _ in sorted(tot.items(), key=lambda x: (-x[1], x[0]))]


# ------------------------------------------------------------------ metrics
def first_rank(ranked: list[str], gold: set[int], page_of: Callable[[str], int | None]) -> int | None:
    for i, c in enumerate(ranked, 1):
        if page_of(c) in gold: return i
    return None


def agg(ranks: list[int | None], ks=KS) -> dict:
    n = len(ranks)
    if not n: return {"n": 0}
    d = {"n": n, "mrr": round(sum(1 / r for r in ranks if r and r <= 40) / n, 4)}
    for k in ks: d[f"R@{k}"] = round(sum(1 for r in ranks if r and r <= k) / n, 4)
    return d


# ------------------------------------------------------------------ context
@dataclass
class Ctx:
    nodes: dict
    ids: list[str]
    dense: object                       # llama-index retriever (similarity_top_k=40)
    gloss: dict
    bm25: dict = field(default_factory=dict)       # name -> (BM25, tokenizer fn)

    def page_of(self, c: str):
        return (self.nodes.get(c) or {}).get("printed_page_number")


def make_ctx(index, gloss) -> Ctx:
    import bm25s
    texts = [index.nodes[i]["text"] for i in index.ids]
    c = Ctx(index.nodes, list(index.ids), index.vector.as_retriever(similarity_top_k=conf.TOP_DENSE), gloss)
    for name, fn in TOKENIZERS.items():
        bm = bm25s.BM25(k1=conf.BM25_K1, b=conf.BM25_B)
        bm.index([fn(t) for t in texts], show_progress=False)
        c.bm25[name] = (bm, fn)
    return c


def bm25_rank(c: Ctx, q: str, tok: str, top: int = conf.TOP_BM25) -> list[tuple[str, float]]:
    bm, fn = c.bm25[tok]
    toks = fn(q)
    if not toks: return []
    sc = np.asarray(bm.get_scores(toks))
    order = np.argsort(-sc, kind="stable")[:top]
    return [(c.ids[i], float(sc[i])) for i in order if sc[i] > 0]


def dense_rank(c: Ctx, q: str) -> list[tuple[str, float]]:
    return [(h.node.node_id, float(h.score or 0.0)) for h in c.dense.retrieve(q)]


# ------------------------------------------------------------------ runner
def _gold(it: ev.Item) -> set[int]:
    return set(it.gold_printed_pages) | set(it.alt_pages)


def _by_set(rows: list[dict], fn) -> dict:
    out = {}
    for s in sorted({r["set"] for r in rows}):
        out[s] = fn([r for r in rows if r["set"] == s])
    out["all"] = fn(rows)
    return out


def run_exp(set_paths: list[str], ctx: Ctx, which: tuple[str, ...] = EXPS, include_unverified: bool = False,
            progress: Callable[[str], None] = lambda s: None) -> dict:
    items: list[tuple[str, ev.Item]] = []
    for p in set_paths:
        for it in ev.load_items(p):
            if it.verified or include_unverified:
                items.append((Path(p).stem, it))
    items = [(s, it.model_copy(update={"id": f"{s}:{it.id}"})) for s, it in items]      # ids are only unique per file
    ans = [(s, it) for s, it in items if it.answerable and _gold(it)]
    una = [(s, it) for s, it in items if not it.answerable]
    if not ans:
        raise ValueError("no verified answerable items with gold pages (verify the dev/val sets or pass --include-unverified)")
    res: dict = {"meta": {"n_answerable": len(ans), "n_unanswerable": len(una), "sets": sorted({s for s, _ in items}),
                          "include_unverified": include_unverified, "experiments": list(which), "device": os.getenv("TCB_DEVICE", "cpu")}, "decisions": {}}
    expq = lambda q: find.expand(q, ctx.gloss)

    # ---- first stage per answerable item (cached)
    fs: dict[str, dict] = {}
    for s, it in ans + una:
        q = it.question
        fs[it.id] = {"set": s, "gold": _gold(it), "q": q, "qx": expq(q),
                     "bm": {t: bm25_rank(ctx, q, t) for t in TOKENIZERS} if "E2" in which else {},
                     "bmx": bm25_rank(ctx, expq(q), "vi"), "dn": dense_rank(ctx, q), "dnx": dense_rank(ctx, expq(q))}
    rows = lambda: [{"set": fs[it.id]["set"], "id": it.id} for _, it in ans]

    def table(fn, ks=KS):
        r = rows()
        return _by_set(r, lambda sub: agg([fn(x["id"]) for x in sub], ks))

    rank = lambda lst, gold: first_rank(lst, gold, ctx.page_of)

    if "E2" in which:
        res["E2"] = {t: table(lambda i, t=t: rank([c for c, _ in fs[i]["bm"][t]], fs[i]["gold"])) for t in TOKENIZERS}
        r20 = {t: res["E2"][t]["all"].get("R@20", 0) for t in TOKENIZERS}
        best = max(TOKENIZERS, key=lambda t: (r20[t], t == "vi"))
        switch = best != "vi" and r20[best] >= r20["vi"] + 0.02          # a rebuild needs a real margin, not one item
        res["decisions"]["E2"] = {"best_bm25_only_R@20": best, "production": "vi", "R@20": r20,
                                  "action": f"switch tokenizer to '{best}' (bump conf.TOKENIZER, rebuild index)" if switch else "none (keep 'vi'; switching needs >= 2 pts R@20)"}

    if "E3" in which:
        e3 = {"bm25": table(lambda i: rank([c for c, _ in fs[i]["bmx"]], fs[i]["gold"])),
              "dense": table(lambda i: rank([c for c, _ in fs[i]["dnx"]], fs[i]["gold"])),
              "rrf60": table(lambda i: rank(fuse_rrf([fs[i]["bmx"], fs[i]["dnx"]], 60), fs[i]["gold"])),
              "rsf": table(lambda i: rank(fuse_rsf([fs[i]["bmx"], fs[i]["dnx"]]), fs[i]["gold"]))}
        res["E3"] = e3
        r20 = {k: v["all"].get("R@20", 0) for k, v in e3.items()}
        winner = max(r20, key=r20.get)
        keep = r20["rrf60"] >= r20[winner] - 0.05
        res["decisions"]["E3"] = {"R@20": r20, "keep": "rrf60" if keep else winner,
                                  "rule": "keep RRF unless another method is >= 5 pts better at R@20"}

    if "E3b" in which:
        e3b = {str(k): table(lambda i, k=k: rank(fuse_rrf([fs[i]["bmx"], fs[i]["dnx"]], k), fs[i]["gold"])) for k in RRF_KS}
        res["E3b"] = e3b
        better = []
        for k in (10, 30, 100):
            sets = [s for s in e3b[str(k)] if s != "all"]
            if sets and all(e3b[str(k)][s].get("R@20", 0) - e3b["60"][s].get("R@20", 0) >= 0.05 for s in sets):
                better.append(k)
        res["decisions"]["E3b"] = {"keep_k": 60 if not better else better[0], "beats_60_by_5pts_on_every_set": better,
                                   "rule": "keep 60 unless >= 5 pts R@20 on dev AND val (every set provided)",
                                   "note": "with a single set the rule cannot confirm 'dev and val' separately"}

    fused = lambda i: fuse_rrf([fs[i]["bmx"], fs[i]["dnx"]], conf.RRF_K)

    if "E4" in which:
        e4 = table(lambda i: rank(fused(i), fs[i]["gold"]), FINAL_KS)
        res["E4"] = e4
        r = {k: e4["all"][f"R@{k}"] for k in FINAL_KS}
        top = max(r.values())
        res["decisions"]["E4"] = {"recall_at_k": r, "choose_k": min(k for k in FINAL_KS if r[k] >= top - 0.02),
                                  "current": conf.FINAL_EVIDENCE,
                                  "rule": "smallest k within 2 pts of the best recall (k blocks go to the LLM; more k = longer prompt)"}

    if "E5" in which:
        k = conf.FINAL_EVIDENCE
        on = {it.id: rank(fused(it.id)[:k], fs[it.id]["gold"]) for _, it in ans}
        off = {it.id: rank(fuse_rrf([bm25_rank(ctx, fs[it.id]["q"], "vi"), fs[it.id]["dn"]], conf.RRF_K)[:k], fs[it.id]["gold"])
               for _, it in ans}
        rr = rows()
        res["E5"] = {"expansion_on": _by_set(rr, lambda sub: agg([on[x["id"]] for x in sub], (5, k))),
                     "expansion_off": _by_set(rr, lambda sub: agg([off[x["id"]] for x in sub], (5, k)))}
        reg = [i for i in on if off[i] is not None and on[i] is None]
        gain = [i for i in on if off[i] is None and on[i] is not None]
        res["decisions"]["E5"] = {"regressions": reg, "gains": gain, "keep_expansion": not reg,
                                  "rule": f"keep if no regressions (gold lost from the top-{k} only when expansion is on)"}

    if "E8" in which:                                      # KPI label named in the question as a third RRF ranking
        k, labels = conf.FINAL_EVIDENCE, find.kpi_labels(ctx.nodes)
        with_lab = lambda i: find.rrf([[c for c, _ in fs[i]["bmx"]], [c for c, _ in fs[i]["dnx"]], find.label_matches(fs[i]["qx"], labels)], conf.RRF_K)
        on = {it.id: rank([c for c, _ in with_lab(it.id)][:k], fs[it.id]["gold"]) for _, it in ans}
        off = {it.id: rank(fused(it.id)[:k], fs[it.id]["gold"]) for _, it in ans}
        rr = rows()
        res["E8"] = {"label_on": _by_set(rr, lambda sub: agg([on[x["id"]] for x in sub], (3, 5, k))),
                     "label_off": _by_set(rr, lambda sub: agg([off[x["id"]] for x in sub], (3, 5, k)))}
        reg = [i for i in on if off[i] is not None and (on[i] is None or on[i] > k)]
        gain = [i for i in on if (off[i] is None or off[i] > k) and on[i] is not None and on[i] <= k]
        res["decisions"]["E8"] = {"regressions": reg, "gains": gain, "keep_label_ranking": not reg,
                                  "rule": f"keep if no gold page leaves the top-{k} when label ranking is on"}
    return res


# ------------------------------------------------------------------ output
def to_markdown(res: dict) -> str:
    L = [f"# Retrieval experiments", "", f"answerable={res['meta']['n_answerable']} unanswerable={res['meta']['n_unanswerable']} sets={res['meta']['sets']}", ""]

    def tbl(title, d, cols=("R@5", "R@8", "R@20", "R@40", "mrr")):
        L.append(f"## {title}"); L.append("| variant | set | n | " + " | ".join(cols) + " |"); L.append("|---|---|---|" + "---|" * len(cols))
        for k, sets in d.items():
            for s, m in sets.items():
                L.append(f"| {k} | {s} | {m.get('n')} | " + " | ".join(str(m.get(c, '-')) for c in cols) + " |")
        L.append("")
    for key, title in (("E2", "E2 BM25-only tokenizer"), ("E3", "E3 first-stage fusion"), ("E3b", "E3b RRF k")):
        if key in res: tbl(title, res[key])
    if "E4" in res: tbl("E4 final k of the fused list", {"fused": res["E4"]}, tuple(f"R@{k}" for k in FINAL_KS))
    if "E5" in res: tbl("E5 glossary expansion (final top-k)", res["E5"], ("R@5", f"R@{conf.FINAL_EVIDENCE}", "mrr"))
    if "E8" in res: tbl("E8 KPI label ranking (final top-k)", res["E8"], ("R@3", "R@5", f"R@{conf.FINAL_EVIDENCE}", "mrr"))
    L.append("## Suggested decisions (not applied)"); L.append("```json"); L.append(json.dumps(res["decisions"], ensure_ascii=False, indent=1)); L.append("```")
    return "\n".join(L) + "\n"


def write(res: dict, out_dir: str, label: str) -> Path:
    d = Path(out_dir) / label
    d.mkdir(parents=True, exist_ok=True)
    (d / "exp.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=list), encoding="utf-8", newline="\n")
    (d / "exp.md").write_text(to_markdown(res), encoding="utf-8", newline="\n")
    return d

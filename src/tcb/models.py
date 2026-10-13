"""Docker build step: fetch the fixed embedding model (BGE-M3) at its pinned revision into /models, never at runtime.

  python -m tcb.models download   # needs huggingface_hub + network (build stage only)
"""
from __future__ import annotations

import sys

from . import conf

SPECS = (
    ("bge-m3", conf.EMBED_MODEL, conf.EMBED_REV, ["onnx/*", "imgs/*", "*.jpg", "colbert_linear.pt", "sparse_linear.pt"]),
)


def download() -> None:
    from huggingface_hub import snapshot_download
    for sub, repo, rev, ignore in SPECS:
        d = conf.MODELS_DIR / sub
        snapshot_download(repo_id=repo, revision=rev, local_dir=str(d), ignore_patterns=ignore)
        (d / ".rev").write_text(rev, encoding="utf-8")
        print(f"ok {repo}@{rev[:7]} -> {d}")


if __name__ == "__main__":
    {"download": download}[sys.argv[1]]()

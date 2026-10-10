"""Docker build step: fetch the two fixed models at pinned revisions into /models, never at runtime.

  python -m tcb.models download   # needs huggingface_hub + network (build stage only)
  python -m tcb.models convert    # bge-m3 ships pytorch_model.bin only -> write model.safetensors (needs torch)
"""
from __future__ import annotations

import sys
from pathlib import Path

from . import conf

SPECS = (
    ("bge-m3", conf.EMBED_MODEL, conf.EMBED_REV, ["onnx/*", "imgs/*", "*.jpg", "colbert_linear.pt", "sparse_linear.pt"]),
    ("bge-reranker-v2-m3", conf.RERANK_MODEL, conf.RERANK_REV, ["assets/*", "*.jpg", "*.png"]),
)


def download() -> None:
    from huggingface_hub import snapshot_download
    for sub, repo, rev, ignore in SPECS:
        d = conf.MODELS_DIR / sub
        snapshot_download(repo_id=repo, revision=rev, local_dir=str(d), ignore_patterns=ignore)
        (d / ".rev").write_text(rev)
        print(f"ok {repo}@{rev[:7]} -> {d}")


def convert() -> None:
    """Weights are byte-identical tensors; only the container format changes (.bin pickle -> safetensors)."""
    import torch
    from safetensors.torch import save_file
    d = conf.MODELS_DIR / "bge-m3"
    binf, st = d / "pytorch_model.bin", d / "model.safetensors"
    if st.exists() or not binf.exists():
        print("convert: nothing to do"); return
    sd = torch.load(binf, map_location="cpu", weights_only=True)
    save_file({k: v.contiguous() for k, v in sd.items()}, str(st), metadata={"format": "pt"})
    binf.unlink()
    print(f"converted {len(sd)} tensors -> {st}")


if __name__ == "__main__":
    {"download": download, "convert": convert}[sys.argv[1]]()

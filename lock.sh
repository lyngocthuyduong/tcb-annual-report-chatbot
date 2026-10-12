#!/usr/bin/env bash
# Regenerate requirements.lock from pyproject.toml (run on a networked machine; needs `pip install uv`).
# Hash-pinned, Linux x86_64 / py3.12, WITHOUT torch + CUDA (Docker installs CPU torch separately).
set -euo pipefail
cd "$(dirname "$0")"
SKIP=""
for p in torch triton cuda-bindings cuda-pathfinder cuda-toolkit nvidia-cublas nvidia-cuda-cupti nvidia-cuda-nvrtc \
         nvidia-cuda-runtime nvidia-cudnn-cu13 nvidia-cufft nvidia-cufile nvidia-curand nvidia-cusolver nvidia-cusparse \
         nvidia-cusparselt-cu13 nvidia-nccl-cu13 nvidia-nvjitlink nvidia-nvshmem-cu13 nvidia-nvtx; do SKIP="$SKIP --no-emit-package $p"; done
uv pip compile pyproject.toml --extra dev --python-version 3.12 --python-platform x86_64-manylinux_2_28 --generate-hashes $SKIP -o requirements.lock
if grep -qiE '^(nvidia|triton|cuda|torch)' requirements.lock; then
  echo "ERROR: GPU/torch packages leaked into requirements.lock; update the SKIP list in lock.sh" >&2; exit 1
fi
echo "lock written: requirements.lock"

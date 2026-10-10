# syntax=docker/dockerfile:1
# NOT BUILT in the authoring sandbox (Docker/PyTorch/Hugging Face unreachable there). First real build = first proof.
# Stage 1: models at pinned revisions (revisions live in src/tcb/conf.py, single source of truth)
FROM python:3.12-slim-bookworm AS models
RUN pip install --no-cache-dir "huggingface_hub==1.33.0"
COPY src/tcb/__init__.py src/tcb/conf.py src/tcb/models.py /w/tcb/
RUN PYTHONPATH=/w python -m tcb.models download

# Stage 2: runtime. Dependencies come from the hash-pinned requirements.lock (see lock.sh); torch (CPU) is installed first.
FROM python:3.12-slim-bookworm
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
RUN apt-get update && apt-get install -y --no-install-recommends poppler-utils \
    && rm -rf /var/lib/apt/lists/* && useradd -m -u 1000 app
WORKDIR /app
RUN pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
COPY requirements.lock ./
RUN pip install --require-hashes --no-deps -r requirements.lock
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-deps .
COPY --from=models --chown=app:app /models /models
# No chown RUN and no .bin->safetensors convert here: each would copy the 4.5 GB of weights into a new layer (a Docker build ran out of disk
# exporting layers). torch 2.14 can load .bin; if `tcb check` ever refuses it, run `python -m tcb.models convert` as a fallback. UNVERIFIED until the first real `tcb check`.
COPY tests ./tests
RUN mkdir -p out idx data && chown -R app:app /app
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/health', timeout=4)" || exit 1
# Fails closed (exit 3) when ./idx is missing/corrupt/incompatible, exit 2 on LLM config errors. No auto-rebuild, no fallback.
CMD ["tcb", "serve"]

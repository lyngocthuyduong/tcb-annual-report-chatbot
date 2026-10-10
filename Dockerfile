# syntax=docker/dockerfile:1
# Stage 1: models at pinned revisions (revisions live in src/tcb/conf.py, single source of truth)
FROM python:3.12-slim-bookworm AS models
RUN pip install --no-cache-dir "huggingface_hub>=0.30"
COPY src/tcb/__init__.py src/tcb/conf.py src/tcb/models.py /w/tcb/
RUN PYTHONPATH=/w python -m tcb.models download

# Stage 2: runtime
FROM python:3.12-slim-bookworm
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu
RUN apt-get update && apt-get install -y --no-install-recommends poppler-utils \
    && rm -rf /var/lib/apt/lists/* && useradd -m -u 1000 app
WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir . pytest
COPY --from=models /models /models
RUN python -m tcb.models convert && chown -R app:app /models
COPY tests ./tests
RUN mkdir -p out idx data && chown -R app:app /app
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/health', timeout=4)" || exit 1
# API arrives in a later step; until then the default command validates the shipped index.
CMD ["tcb", "validate", "--snapshot"]

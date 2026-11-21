# syntax=docker/dockerfile:1
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # app configuration (override in compose)
    VLM_BACKEND=auto \
    VLM_MODEL_ID=HuggingFaceTB/SmolVLM-256M-Instruct \
    MODEL_CACHE_DIR=/models

WORKDIR /srv/app

# CPU-only torch first (separate index), then the rest
RUN pip install --no-cache-dir torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY scripts ./scripts

# Optionally bake model weights into the image so the container runs fully
# offline. Build with: docker build --build-arg PRELOAD_MODEL=true .
# (compose sets this by default; requires network access to a HF endpoint)
ARG PRELOAD_MODEL=false
ARG HF_ENDPOINT=""
RUN if [ "$PRELOAD_MODEL" = "true" ]; then \
        if [ -n "$HF_ENDPOINT" ]; then export HF_ENDPOINT="$HF_ENDPOINT"; fi && \
        python scripts/preload_model.py ; \
    fi

RUN useradd --create-home appuser && mkdir -p /models && chown -R appuser /models /srv/app
USER appuser
VOLUME ["/models"]

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=300s --retries=3 \
    CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=4)" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

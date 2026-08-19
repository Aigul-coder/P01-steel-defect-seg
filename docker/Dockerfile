# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    INFERENCE_BACKEND=torch \
    DEVICE=cpu \
    WEIGHTS_PATH=/app/artifacts/model.pt \
    ONNX_PATH=/app/artifacts/model.onnx

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src ./src

# Optional prefetched wheels (local flaky networks). CI usually installs from index.
COPY .wheels/ /tmp/wheels/

RUN pip install --upgrade pip \
    && if ls /tmp/wheels/torch*cpu*manylinux*.whl 1>/dev/null 2>&1; then \
         pip install /tmp/wheels/torch*cpu*manylinux*.whl /tmp/wheels/torchvision*cpu*manylinux*.whl; \
       else \
         pip install --retries 20 torch torchvision --index-url https://download.pytorch.org/whl/cpu \
         || pip install --retries 20 torch torchvision; \
       fi \
    && pip install --retries 10 . \
    && rm -rf /tmp/wheels

COPY configs ./configs
COPY scripts ./scripts
COPY artifacts/metrics.example.json ./artifacts/metrics.example.json

RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/artifacts \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

CMD ["uvicorn", "steel_defect.api:app", "--host", "0.0.0.0", "--port", "8000"]

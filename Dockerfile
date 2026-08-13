FROM python:3.11-slim

ARG VCS_REF=unknown

LABEL org.opencontainers.image.title="Cats vs Dogs MLOps API" \
      org.opencontainers.image.description="FastAPI inference service for the PyTorch cats-versus-dogs classifier" \
      org.opencontainers.image.revision="${VCS_REF}"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_PATH=/app/models/resnet50_baseline.pt \
    FEEDBACK_PATH=/app/runtime/feedback.csv

WORKDIR /app

# libgomp1 is required by the CPU PyTorch runtime on slim Debian images.
RUN apt-get update \
    && apt-get install --no-install-recommends --yes libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-api.txt ./requirements-api.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install \
        --index-url https://download.pytorch.org/whl/cpu \
        torch==2.4.1 \
        torchvision==0.19.1 \
    && python -m pip install --requirement requirements-api.txt

RUN groupadd --system app \
    && useradd --system --gid app --create-home app \
    && mkdir --parents /app/runtime \
    && chown app:app /app/runtime

COPY --chown=app:app src/cats_dogs_mlops ./cats_dogs_mlops
COPY --chown=app:app models/resnet50_baseline.pt ./models/resnet50_baseline.pt

USER app

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=4 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=5).read()"]

CMD ["python", "-m", "uvicorn", "cats_dogs_mlops.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

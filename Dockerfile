# syntax=docker/dockerfile:1.7
# Image for the autonomous lab agent stack.
#
# Built on top of pytorch/pytorch which ships Python + torch + CUDA
# pre-installed (no 3+ GB of nvidia-* wheel downloads during build).
# `docker compose build` produces a local `biolab-agent-base:latest`
# image; this image is NOT published to Docker Hub.

ARG PYTORCH_TAG=2.4.1-cuda12.4-cudnn9

# Stage 1: install apt + Python deps into the pre-existing pytorch env.
FROM pytorch/pytorch:${PYTORCH_TAG}-devel AS build

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.4.30 /uv /uvx /usr/local/bin/

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY eval ./eval

# Install into the container's existing Python env (the pytorch image
# sets up Python at /opt/conda/bin/python). uv --system + UV_SYSTEM_PYTHON=1
# keeps us out of venv territory; torch is already present and satisfies
# the project's ">=2.4,<3" bound, so it is not reinstalled.
ENV UV_SYSTEM_PYTHON=1 \
    UV_HTTP_TIMEOUT=600
RUN uv pip install --system --no-cache -e ".[ui]"

# Stage 2: runtime  -  same pytorch base (runtime variant), copy Python env + source.
FROM pytorch/pytorch:${PYTORCH_TAG}-runtime AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    BIOLAB_DATA_DIR=/data \
    BIOLAB_ARTIFACT_DIR=/artifacts \
    OLLAMA_HOST=http://ollama:11434 \
    QDRANT_URL=http://qdrant:6333

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy over the installed site-packages from the build stage.
COPY --from=build /opt/conda /opt/conda

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY eval ./eval
COPY scripts ./scripts
COPY tests ./tests
COPY configs ./configs
COPY data ./data

RUN mkdir -p /data /artifacts /logs \
    && chmod +x scripts/*.sh

EXPOSE 8000 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -fsS http://localhost:8000/healthz || exit 1

CMD ["uvicorn", "biolab_agent.server:app", "--host", "0.0.0.0", "--port", "8000"]

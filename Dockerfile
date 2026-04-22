# ── Stage 1: Builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Build deps needed to compile some ML packages
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
# Use the CPU-only index to keep the image size under control
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# libgomp1 is required by LightGBM/XGBoost at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face requires UID 1000
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Copy the venv from builder
COPY --from=builder /opt/venv /opt/venv

# CRITICAL: Copy all project folders including models
# Hugging Face does not support docker-compose mounts for artifacts
COPY --chown=user api/     ./api/
COPY --chown=user src/     ./src/
COPY --chown=user config/  ./config/
COPY --chown=user app/     ./app/
COPY --chown=user models/  ./models/

# Hugging Face Spaces port is ALWAYS 7860
EXPOSE 7860

# Run with the background-loading lifespan we built to prevent timeouts
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
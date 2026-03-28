# Dockerfile — Fraud Detection API (optimised)
#
# Optimisations applied:
#   1. Multi-stage build   — builder stage discarded, only runtime ships
#   2. python:3.12-slim    — 150 MB base vs 900 MB full Python image
#   3. --no-cache-dir      — pip doesn't store download cache in the layer
#   4. Chained RUN + clean — single layer for install + cleanup
#   5. requirements.txt first — layer caching: pip only re-runs when deps change
#   6. .dockerignore        — venv/, tests/, *.pkl excluded from build context
#

# ── Stage 1: builder ─────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /app

COPY requirements.txt .

# Chain install + cache cleanup in ONE RUN → single layer, no cache residue
RUN pip install --no-cache-dir -r requirements.txt \
    && find /usr -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true

COPY . .
RUN python train_and_save_model.py

# ── Stage 2: runtime ─────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

WORKDIR /app

ENV MODEL_PATH=/app/fraud_model.pkl
ENV FRAUD_THRESHOLD=0.5
ENV LOG_LEVEL=info

COPY requirements.txt .

# Same pattern: install + clean in one RUN
RUN pip install --no-cache-dir -r requirements.txt \
    && find /usr -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true

# Copy ONLY the two files needed to serve predictions
COPY --from=builder /app/main.py         ./main.py
COPY --from=builder /app/fraud_model.pkl ./fraud_model.pkl

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
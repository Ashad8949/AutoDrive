# ── Stage 1: Build / dependency install ──────────────
FROM python:3.12-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Stage 2: Runtime (minimal image) ─────────────────
FROM python:3.12-slim
WORKDIR /app

COPY --from=builder /install /usr/local
COPY src/ ./src/
COPY seed_data.json .

ENV PYTHONPATH=/app
EXPOSE 8002

CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8002", "--workers", "2"]

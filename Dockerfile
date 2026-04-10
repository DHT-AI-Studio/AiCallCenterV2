# Stage 1: Builder
FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04 AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    build-essential \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app
COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-dev

# Stage 2: Runtime
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04 AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.12 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.12 /usr/local/bin/python3

WORKDIR /app
COPY --from=builder /root/.local/bin/uv /usr/local/bin/uv
COPY --from=builder /app/.venv /app/.venv

ENV PATH="/app/.venv/bin:$PATH"
ENV VIRTUAL_ENV="/app/.venv"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV LD_LIBRARY_PATH="/app/.venv/lib/python3.12/site-packages/nvidia/cusparselt/lib:${LD_LIBRARY_PATH}"

COPY . .
# COPY voices/ /app/voices/
COPY output/transcode/greeting.wav /app/output/transcode/greeting.wav
RUN mkdir -p /app/output/transcode /app/output/response /app/output/converted /app/recording /app/voices

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import webrtcvad; import websockets; import pydantic; print('OK')" || exit 1

EXPOSE 5060/udp 5062/udp 8080/tcp

CMD ["python", "receive_server.py"]
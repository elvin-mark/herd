# --- Stage 1: Build llama-server and whisper-server binaries ---
FROM python:3.10-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    libcurl4-openssl-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Clone and build llama.cpp with libcurl support (for downloading GGUF files directly)
RUN git clone --depth 1 https://github.com/ggerganov/llama.cpp.git && \
    cd llama.cpp && \
    mkdir build && \
    cd build && \
    cmake .. -DLLAMA_CURL=ON && \
    cmake --build . --config Release --target llama-server

# Clone and build whisper.cpp
RUN git clone --depth 1 https://github.com/ggerganov/whisper.cpp.git && \
    cd whisper.cpp && \
    mkdir build && \
    cd build && \
    cmake .. && \
    cmake --build . --config Release --target whisper-server


# --- Stage 2: Final lightweight image ---
FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libcurl4 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy compiled binaries into execution path
COPY --from=builder /build/llama.cpp/build/bin/llama-server /usr/local/bin/llama-server
COPY --from=builder /build/whisper.cpp/build/bin/whisper-server /usr/local/bin/whisper-server

# Pre-verify binaries are available
RUN llama-server --help > /dev/null 2>&1 || true
RUN whisper-server --help > /dev/null 2>&1 || true

# Copy project files and install Herd CLI package
COPY pyproject.toml README.md ./
COPY src/herd/ ./src/herd/
COPY assets/ ./assets/

RUN pip install --no-cache-dir .

# Default configuration settings
ENV HERD_HOST=0.0.0.0
ENV HERD_PORT=11434
ENV HERD_HOME=/root/.herd

# Expose Herd API Gateway port
EXPOSE 11434

# Start the gateway
ENTRYPOINT ["herd", "serve"]

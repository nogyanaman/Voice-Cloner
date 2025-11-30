# Base image: CUDA 12.4 + Ubuntu 22.04 (stable)
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# ----------------------------------------------------
# System packages
# ----------------------------------------------------
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ----------------------------------------------------
# Copy requirements into image
# ----------------------------------------------------
COPY requirements.txt /app/requirements.txt

# ----------------------------------------------------
# IMPORTANT: Fix pip version for fairseq/omegaconf
# ----------------------------------------------------
RUN python3 -m pip install --upgrade pip && pip install "pip<24.1"

# ----------------------------------------------------
# Install PyTorch for Linux CUDA (cu121)
# ----------------------------------------------------
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# ----------------------------------------------------
# Install project Python dependencies
# ----------------------------------------------------
RUN pip install -r requirements.txt

# ----------------------------------------------------
# Copy project source code
# ----------------------------------------------------
COPY src /app/src
COPY models /app/models
COPY data /app/data
COPY config.yaml /app/
COPY pyproject.toml /app/
COPY Makefile /app/

ENV PYTHONPATH=/app

# ----------------------------------------------------
# Default command
# ----------------------------------------------------
ENTRYPOINT ["python3", "-m", "voice_conv.cli"]

# ===========================================
# AGENTIC RAG - Production Dockerfile
# GPU (CUDA 12.1) | Python 3.11
# ===========================================

FROM nvidia/cuda:12.1.0-base-ubuntu22.04

# Sistem değişkenleri
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

# Sistem bağımlılıkları
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Çalışma dizini
WORKDIR /app

# Requirements kopyalama ve yükleme
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Uygulama dosyaları
COPY . .

# PYTHONPATH ayarı (ÖNEMLİ!)
ENV PYTHONPATH=/app

# HuggingFace cache dizini
ENV HF_HOME=/app/.cache/huggingface
RUN mkdir -p /app/.cache/huggingface

# Port
EXPOSE 7860

# Başlangıç komutu
CMD ["python3", "app/main.py"]
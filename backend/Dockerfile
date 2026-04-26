FROM python:3.11-slim

# -----------------------------
# ENV SETTINGS (better defaults)
# -----------------------------
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory early
WORKDIR /app

# -----------------------------
# SYSTEM DEPENDENCIES
# -----------------------------
RUN apt-get update \
  && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libmagic1 \
    ffmpeg \
    libsndfile1 \
    libpango-1.0-0 \
    libpangoft2-1.0-0 \
    libpangocairo-1.0-0 \
    libgdk-pixbuf-xlib-2.0-0 \
    libffi-dev \
    shared-mime-info \
    gcc \
    g++ \
  && rm -rf /var/lib/apt/lists/*

# -----------------------------
# PYTHON DEPENDENCIES
# -----------------------------
COPY requirements.txt ./

# Upgrade pip (important for mediapipe + wheels)
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# -----------------------------
# MODEL CACHE (HuggingFace / Torch)
# -----------------------------
ENV HF_HOME=/app/models
ENV TORCH_HOME=/app/models

# Create model cache dir (prevents runtime issues)
RUN mkdir -p /app/models

# -----------------------------
# APP CODE
# -----------------------------
COPY . .

# -----------------------------
# PORT
# -----------------------------
EXPOSE 8000

# -----------------------------
# START SERVER
# -----------------------------
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
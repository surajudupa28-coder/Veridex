FROM python:3.11-slim


# System dependencies for OpenCV (headless), audio processing, and file-type detection
# - libgl1/libglib2.0-0/libsm6/libxrender1/libxext6: OpenCV runtime deps
# - libmagic1: python-magic backend
# - ffmpeg/libsndfile1: audio decoding + soundfile/librosa support
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
  && rm -rf /var/lib/apt/lists/*

# Set the application working directory
WORKDIR /app

# Install Python dependencies first (better layer caching)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Cache HuggingFace + Torch models under a mountable directory
ENV HF_HOME=/app/models
ENV TORCH_HOME=/app/models

# Copy the rest of the application code
COPY . .

# FastAPI default port
EXPOSE 8000

# Start the API server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
>>>>>>> 28c2acc (Initial Veridex file setup and image detector updated)

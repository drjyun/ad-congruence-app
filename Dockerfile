# Use Python base with optimized package installation
FROM python:3.10-slim

# Install system packages first (for caching)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Install TensorFlow FIRST (biggest package - ~500MB)
RUN pip install --no-cache-dir tensorflow==2.15.0

# Install PyTorch (second biggest - CPU-only version is much smaller)
RUN pip install --no-cache-dir \
    torch==2.2.2 \
    torchvision==0.17.2 \
    --index-url https://download.pytorch.org/whl/cpu

# Install TensorFlow Hub and Transformers
RUN pip install --no-cache-dir \
    tensorflow-hub==0.16.1 \
    transformers==4.44.2

# Install remaining lightweight packages
RUN pip install --no-cache-dir \
    gradio==4.44.0 \
    librosa==0.10.2.post1 \
    soundfile==0.12.1 \
    matplotlib==3.9.2 \
    pandas==2.2.2 \
    pillow==10.4.0 \
    requests==2.32.3

WORKDIR /app

# Copy only necessary files (exclude large files)
COPY app.py requirements.txt ./
COPY audio/ ./audio/
COPY visual/ ./visual/

# Set environment variables
ENV PORT=7860 \
    PYTHONUNBUFFERED=1 \
    TFHUB_CACHE_DIR=/tmp/tfhub_cache \
    TRANSFORMERS_CACHE=/tmp/transformers_cache

EXPOSE 7860

CMD ["python", "app.py"]


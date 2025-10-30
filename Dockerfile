# Use TensorFlow base image (already has TensorFlow + dependencies)
FROM tensorflow/tensorflow:2.15.0-py3

# Install PyTorch and other ML libraries (faster than from scratch)
RUN pip install --no-cache-dir \
    torch==2.2.2 \
    torchvision==0.17.2 \
    transformers==4.44.2 \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining lightweight packages
RUN pip install --no-cache-dir \
    gradio==4.44.0 \
    librosa==0.10.2.post1 \
    soundfile==0.12.1 \
    matplotlib==3.9.2 \
    pandas==2.2.2 \
    pillow==10.4.0 \
    requests==2.32.3

# Install system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only necessary files (exclude large files)
COPY app.py requirements.txt ./
COPY audio/ ./audio/
COPY visual/ ./visual/

ENV PORT=8080 \
    PYTHONUNBUFFERED=1

EXPOSE 8080

CMD ["python", "app.py"]


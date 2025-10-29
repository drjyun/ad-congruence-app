FROM python:3.11-slim

# Install ffmpeg for audio extraction
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Set default port for Render
ENV PORT=8000
CMD ["uvicorn", "ad_context_congruence_api_fast_api_backend:app", "--host", "0.0.0.0", "--port", "8000"]

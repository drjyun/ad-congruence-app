FROM python:3.10-slim

# (Optional) system libs if you ever add OpenCV/ffmpeg; safe to omit otherwise
# RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libsndfile1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .
# Streamlit
CMD ["bash","-lc","streamlit run app.py --server.port $PORT --server.address 0.0.0.0"]

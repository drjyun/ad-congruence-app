FROM python:3.10-slim

# Optional but helpful for smaller images & faster installs
ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install only what's needed for the UI
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Add app code
COPY . .

# Optional: pick a default port so it also runs locally
ENV PORT=8080

# Run Streamlit and bind to Render's provided $PORT
CMD ["bash","-lc","streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port $PORT"]

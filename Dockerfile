FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1

# Install system dependencies for OpenCV & MediaPipe
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libxcb1 \
    libxext6 \
    libxrender1 \
    libsm6 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy và install requirements
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY backend ./backend
COPY code ./code
COPY data ./data
COPY chroma_db_official ./chroma_db_official
COPY video ./video

# Expose Hugging Face Spaces port
EXPOSE 7860

# Run FastAPI
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-7860}"]

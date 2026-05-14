FROM python:3.10-slim

WORKDIR /app

# Install system dependencies with retry logic
RUN apt-get update --fix-missing || apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    gcc \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Upgrade pip first
RUN pip install --upgrade pip setuptools wheel

# Copy requirements
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt || pip install --no-cache-dir -r requirements.txt --retries 5

# Copy source code
COPY backend ./backend
COPY code ./code
COPY data ./data
COPY chroma_db_official ./chroma_db_official
COPY video ./video

# Expose port
EXPOSE 8000

# Run FastAPI
CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
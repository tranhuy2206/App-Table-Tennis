FROM python:3.10

WORKDIR /app

# Upgrade pip first (no system dependencies needed - opencv wheels include binaries)
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

# Expose port
EXPOSE 8000

# Run FastAPI
CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
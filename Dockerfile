FROM python:3.10-alpine

WORKDIR /app

# Alpine cần lệnh khác
RUN apk add --no-cache \
    gcc \
    musl-dev \
    libffi-dev \
    libgl \
    libglib \
    libsm \
    libxext \
    libxrender

RUN pip install --upgrade pip
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend ./backend
COPY code ./code
COPY data ./data
COPY chroma_db_official ./chroma_db_official
COPY video ./video

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
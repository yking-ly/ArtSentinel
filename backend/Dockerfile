# Use a small Python base. If you want official PyTorch CPU image use pytorch/pytorch:2.x-cpu
FROM python:3.10-slim

# avoid buffering logs
ENV PYTHONUNBUFFERED=1
ENV POETRY_VIRTUALENVS_CREATE=false

WORKDIR /app

# system deps for pillow, albumentations; keep it minimal
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# copy requirements and install
COPY backend/requirements.txt /app/requirements.txt
RUN pip install --upgrade pip setuptools wheel
RUN pip install -r /app/requirements.txt

# copy app and model files
COPY backend /app

# ensure model path directory exists
RUN mkdir -p /app/models

# port (Render sets PORT env; use $PORT)
EXPOSE 8000

# Start command (gunicorn + uvicorn worker)
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "app.main:app", "--bind", "0.0.0.0:$PORT", "--workers", "1", "--threads", "4"]

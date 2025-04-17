# Use official Python slim image
FROM python:3.11-slim-bullseye

# Install required OS-level dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libjpeg-dev \
    zlib1g-dev \
    libfreetype6-dev \
    pkg-config \
    libharfbuzz-dev \
    libfribidi-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY pdf-summarizer/requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy project files
COPY . .

# Expose port (Railway will auto-assign, but this is good practice)
EXPOSE 8000

# Run FastAPI with Uvicorn, using Railway's PORT variable
CMD ["sh", "-c", "uvicorn pdf-summarizer.app:app --host 0.0.0.0 --port ${PORT:-8000}"]

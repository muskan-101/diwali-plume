# Stage 1: Build React Frontend
FROM node:18-alpine AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# Stage 2: Python Backend & Production Runtime
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=5000

# Install system dependencies (FFmpeg for video transcoding, OpenCV libgl requirements)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
# Install CPU-optimized PyTorch first to save disk footprint and build fast, then remaining requirements
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy backend application, models, and sample dataset
COPY src/ src/
COPY models/ models/
COPY yolov8n.pt .
COPY data/sample/ data/sample/

# Copy compiled frontend from Stage 1
COPY --from=frontend-builder /app/frontend/dist frontend/dist

# Create runtime directories
RUN mkdir -p data/uploads data/outputs data/results

EXPOSE 5000

# Start unified API & frontend server
CMD ["python", "src/api/app.py", "--port", "5000"]

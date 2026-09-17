FROM python:3.11-slim

WORKDIR /app

# Install minimal system deps (no git-lfs needed — artifacts come from volume mount)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only Python dependencies (no CUDA = much smaller image)
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

# Copy application source code only (artifacts/ excluded via .dockerignore)
COPY src/ ./src/
COPY frontend/ ./frontend/
COPY download_artifacts.py ./

# Environment
ENV PYTHONPATH=src
ENV HOST=0.0.0.0
ENV PORT=7860

# Artifacts are mounted as a volume at runtime:
#   ./artifacts  →  /artifacts  (FAISS index, classifier, cluster_assignments, etc.)
# Nothing from artifacts/ is baked into this image.
EXPOSE 7860

CMD uvicorn research_ai.api.main:app --host 0.0.0.0 --port 7860 --workers 1

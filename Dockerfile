FROM python:3.11-slim

WORKDIR /app

# Install git and git-lfs for downloading LFS artifacts if needed
RUN apt-get update && apt-get install -y git git-lfs && rm -rf /var/lib/apt/lists/*
RUN git lfs install

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Set environment variables
ENV PYTHONPATH=src
ENV HOST=0.0.0.0
ENV PORT=7860

# Expose port
EXPOSE 7860

# Run startup script to download artifacts (if HF_ARTIFACTS_REPO is set)
# Then start the FastAPI server directly
CMD python download_artifacts.py && uvicorn research_ai.api.main:app --host 0.0.0.0 --port 7860 --workers 1

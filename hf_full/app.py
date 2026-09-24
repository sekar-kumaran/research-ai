import os
import sys

# Prepend src to sys.path so internal imports resolve
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

import gradio as gr
from fastapi import FastAPI
from deployment.zero_gpu import ZeroGPUEmbeddingService
from research_ai.api.main import app as fastapi_app, platform

# --- 1. OVERRIDE ML SERVICES FOR ZEROGPU ---
# Initialize the ZeroGPUEmbeddingService
gpu_embedding = ZeroGPUEmbeddingService(
    model_name=platform.settings.retrieval.embedding_model_name,
    device=os.environ.get("EMBEDDING_DEVICE", "auto")
)

# Patch the platform components to use the GPU-enabled embedding service
platform.embedding_service = gpu_embedding
platform.retriever.embedding_service = gpu_embedding
platform.similarity.embedding_service = gpu_embedding
platform.paper_chat.embedding_service = gpu_embedding


# --- 2. GRADIO SERVER MOUNT ---
# Create a dummy Gradio UI block
# Even though this is an API-first backend, HF Gradio Spaces expect a UI or FastAPI instance.
# We mount the Gradio app onto the existing FastAPI app.
demo = gr.Blocks()
with demo:
    gr.Markdown("# Research AI Backend API (ZeroGPU Enabled)")
    gr.Markdown("This space hosts the complete Research AI backend and ML runtime.")
    gr.Markdown("API Documentation: [Swagger UI](/docs)")

# Mount the Gradio blocks onto the FastAPI app
# HF Spaces looks for an ASGI `app` variable when starting the uvicorn server.
app = gr.mount_gradio_app(fastapi_app, demo, path="/ui")

# Expose the FastAPI app so `uvicorn app:app` can run it

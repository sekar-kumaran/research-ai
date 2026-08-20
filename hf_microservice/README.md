# Research AI — ML Microservice

This directory contains the remote ML microservice component for the Research AI Intelligence Platform. It runs the heavy Machine Learning models (semantic search, classification, clustering, summarization) via Hugging Face ZeroGPU.

## Deployment Architecture

The Research AI platform is designed as a **two-process architecture**:

1. **Main Orchestrator (FastAPI)**: The primary web server and LLM agent orchestrator (deployed via `Dockerfile` or `render.yaml` at the repo root).
2. **ML Microservice (Gradio)**: The GPU-accelerated backend that executes local ML models (deployed via `hf_microservice/app.py`).

**You MUST deploy this microservice as a separate Hugging Face Space** using the Gradio SDK.

### Deployment Instructions

1. Create a new Hugging Face Space.
2. Select **Gradio** as the SDK.
3. If available, enable **ZeroGPU** for hardware acceleration.
4. Set the Space to point to this directory (or duplicate the code so `hf_microservice/app.py` is the root `app.py` of the new Space).
5. Add your precomputed ML artifacts to a Hugging Face Dataset repo, and configure `HF_ARTIFACTS_REPO` in the microservice Space settings.

### Connecting the Main App to the Microservice

Once the microservice is deployed and running successfully, copy its Space ID (e.g., `sekarkumaran461/research-ai-ml`).

Go to the settings of your **Main Orchestrator Space** and add the following Environment Variable:
```env
HF_SPACE_ID=sekarkumaran461/research-ai-ml
```

The main app will use the `gradio_client` library to route all ML tasks to this microservice.

### Verifying the Connection

To verify the microservice is live and connected:
1. Ensure the Microservice Space is "Running" on Hugging Face.
2. Visit the `/health` endpoint of your Main Orchestrator app.
3. It will verify connectivity to the ML microservice. If it fails, check the Orchestrator's startup logs.

# Deployment Guide (HF Space Full)

## Repository Target
**Space Name**: `sekarkumaran461/ResearchAi-Full`
**SDK**: Gradio
**Hardware**: ZeroGPU (via Hugging Face `@spaces.GPU` integration)

## Environment Variables
Must be configured in Hugging Face Space Settings -> Variables and Secrets:

**Variables**:
- `LLM_BACKEND=cloud`
- `CLOUD_LLM_PROVIDER=groq` (or `openrouter`, `google`)
- `EMBEDDING_DEVICE=auto` (Enables automatic ZeroGPU offloading)
- `ENABLE_PYTHON_EXECUTION=false` (Security: Disable local script execution)
- `CORS_ORIGINS=https://your-react-domain.vercel.app` (Security)

**Secrets**:
- `GROQ_API_KEY` or `OPENROUTER_API_KEY` depending on chosen provider.

## Storage Architecture
Hugging Face Spaces use ephemeral local disks.
- The `ConversationStore` (Chat History) is currently in-memory. If the Space restarts, active chat context is lost.
- PDF Uploads are temporarily stored in `/tmp/` and successfully cleaned up after parsing.
- Future production hardening requires configuring `DATABASE_URL` to point to a managed Postgres instance (like Neon) to persist conversation state permanently.

## Deployment Steps
1. Verify `hf_full` passes the local test suite `pytest tests/test_api.py`.
2. Initialize git inside `hf_full` or clone the Hugging Face Space directly:
   `git clone https://huggingface.co/spaces/sekarkumaran461/ResearchAi-Full`
3. Copy the contents of `hf_full/` into the cloned repo.
4. Verify no `.env` files or API keys are committed.
5. Push to Hugging Face:
   `git add .`
   `git commit -m "Deploy Research AI full backend"`
   `git push`
6. Wait for the space to build and transition to "Running".
7. Query the `/health` endpoint directly.

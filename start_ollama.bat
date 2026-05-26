@echo off
set CLOUD_LLM_PROVIDER=ollama
set OLLAMA_BASE_URL=http://localhost:11434/v1
set OLLAMA_MODEL=qwen2.5:3b
set LLM_BACKEND=cloud
set DATA_ROOT=data
set ARTIFACTS_ROOT=artifacts
cd /d "d:\main projects\research_ai_platform_v3.1_final"
python -m uvicorn research_ai.api.main:app --host 127.0.0.1 --port 8000

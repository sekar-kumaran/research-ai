@echo off
setlocal

REM Start Ollama and the Research AI FastAPI application together.
REM Keep this file in the repository root and run it with:
REM   start_all_ollama_app.bat

cd /d "%~dp0"

set CLOUD_LLM_PROVIDER=ollama
set OLLAMA_BASE_URL=http://localhost:11434/v1
set OLLAMA_MODEL=qwen2.5:3b
set LLM_BACKEND=cloud
set DATA_ROOT=data
set ARTIFACTS_ROOT=artifacts
set PYTHONPATH=%CD%\src

where ollama >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Ollama is not installed or not available in PATH.
    echo Install Ollama first, then run this file again.
    exit /b 1
)

echo ==========================================
echo   Research AI + Ollama launcher
echo ==========================================
echo   Provider : %CLOUD_LLM_PROVIDER%
echo   Model    : %OLLAMA_MODEL%
echo   Ollama   : %OLLAMA_BASE_URL%
echo   App URL  : http://127.0.0.1:8000
echo ==========================================
echo.

echo [1/4] Starting Ollama server in a background window...
start "Ollama Server" /min cmd /c "ollama serve"

echo [2/4] Waiting for Ollama to become reachable...
timeout /t 5 /nobreak >nul

echo [3/4] Ensuring the model exists locally: %OLLAMA_MODEL%
ollama list | findstr /i /c:"%OLLAMA_MODEL%" >nul
if errorlevel 1 (
    echo Model not found. Pulling %OLLAMA_MODEL% now...
    ollama pull %OLLAMA_MODEL%
    if errorlevel 1 (
        echo [ERROR] Failed to pull %OLLAMA_MODEL%.
        exit /b 1
    )
) else (
    echo Model already available.
)

echo [4/4] Starting Research AI FastAPI application...
echo.
python -m uvicorn research_ai.api.main:app --host 127.0.0.1 --port 8000

endlocal

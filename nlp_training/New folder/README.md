# NLP Research Assistant Project

This folder contains an AI-powered research paper assistant that runs from prebuilt artifacts.

## Implemented Components
- Supervised classification: Naive Bayes, Logistic Regression, Linear SVM
- Unsupervised topic clustering: K-Means
- Similarity search: MiniLM embeddings + FAISS
- Summarization: pretrained transformer pipeline
- Retrieval-augmented QA: embedding retrieval + generator
- API serving: FastAPI

## Quick Start
1. Create and activate virtual environment.
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Ensure artifacts are present (already generated):
   - `artifacts/classification/classifier.joblib`
   - `artifacts/classification/tfidf_vectorizer.joblib`
   - `artifacts/classification/labels.joblib`
   - `artifacts/clustering/kmeans.joblib`
   - `artifacts/similarity/paper_index.faiss`
   - `artifacts/similarity/paper_metadata.parquet`
4. Run API:
   - `uvicorn src.api.main:app --reload --port 8000`

## Inference-Only Project Policy
- Training scripts and training notebooks have been removed from this project copy.
- Runtime uses only saved artifacts for inference/search/summarization.
- If future retraining is needed, it should happen in a separate training workspace, then only artifacts should be copied here.

## Deploy With Free Cloud LLMs
Use cloud inference to avoid downloading large local generation models.

1. Set shared backend mode:
   - `LLM_BACKEND=cloud`

2. Choose provider:
   - `CLOUD_LLM_PROVIDER=groq` or `CLOUD_LLM_PROVIDER=openrouter` or `CLOUD_LLM_PROVIDER=google`

3. For Groq free tier:
   - `GROQ_API_KEY=your_key`
   - Optional: `GROQ_MODEL=llama-3.1-8b-instant`

4. For OpenRouter free tier:
   - `OPENROUTER_API_KEY=your_key`
   - Optional: `OPENROUTER_MODEL=meta-llama/llama-3.1-8b-instruct:free`

5. For Google Gemini:
   - `GOOGLE_API_KEY=your_key`
   - Optional: `GOOGLE_MODEL=gemini-1.5-flash`

6. LangChain agent backend (optional override):
   - `LC_LLM_BACKEND=cloud`

With these env vars, summarization, RAG generation, and LangChain auto-agent run on cloud LLMs.

## Credentials Needed
Only cloud LLM credentials are optional and required if you enable cloud generation:
- `GROQ_API_KEY` for Groq
- `OPENROUTER_API_KEY` for OpenRouter
- `GOOGLE_API_KEY` for Google Gemini

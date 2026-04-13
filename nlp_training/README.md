# 🔬 Arxiv Research AI — Enhanced Pipeline

AI-Powered Research Paper Retrieval, Classification & Summarisation System built on the arXiv dataset.

---

## 🗂 Architecture

```
User (Browser)
    ↓ HTTP
FastAPI Backend  (/src/api/main.py)
    ├── /search         → FAISS Vector Search (all-MiniLM-L6-v2)
    ├── /classify       → TF-IDF + Sklearn Classifier
    ├── /summarize      → Cloud LLM / DistilBART
    ├── /ask            → RAG Pipeline (retrieve → generate)
    ├── /agent/run      → Multi-mode agent dispatcher
    ├── /chat/upload    → PDF upload → FAISS session
    └── /chat/ask       → Per-paper Q&A with history
         ↓
    Cloud LLM (Groq / OpenRouter / Google)
    OR Local (flan-t5, distilbart)
```

---

## ⚡ Quick Start

### 1. Setup environment

```bash
cp .env.example .env
# Edit .env — add your GROQ_API_KEY (free at console.groq.com)
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Build the pipeline artifacts (first time only)

Run these pipeline scripts in order against your arXiv dataset:

```bash
# 1. Preprocess & clean data
python pipeline_scripts/01_clean_preprocess.py

# 2. Build TF-IDF + train classifier
python pipeline_scripts/04_train_models.py

# 3. Build FAISS embedding index
python pipeline_scripts/05_build_embeddings.py   # (if present)
```

### 4. Start the server

```bash
bash start.sh
# OR
uvicorn src.api.main:app --reload
```

Open http://localhost:8000 in your browser.

---

## 🖥 Frontend Features

| Mode | What it does |
|------|-------------|
| **Ask AI** | RAG-powered Q&A over the arXiv index with streaming |
| **Search** | Semantic FAISS search — returns ranked paper cards |
| **Classify** | Predicts arXiv category (cs/math/physics/q-bio) with confidence |
| **Summarize** | Cloud-LLM summary with key contribution, method, findings |
| **Paper Chat** | Upload PDF or load by arXiv ID, then chat with the full paper |

---

## 🔑 Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_BACKEND` | `cloud` | `cloud` or `local` |
| `CLOUD_LLM_PROVIDER` | `groq` | `groq`, `openrouter`, `google` |
| `GROQ_API_KEY` | — | Groq API key (get free at console.groq.com) |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Groq model name |
| `GOOGLE_API_KEY` | — | Google AI API key |
| `OPENROUTER_API_KEY` | — | OpenRouter API key |
| `DATA_ROOT` | `data` | Path to arXiv dataset shards |
| `ARTIFACTS_ROOT` | `artifacts` | Path to trained artifacts |

---

## 📡 API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `GET /health` | GET | Component status |
| `GET /stats` | GET | Index & session stats |
| `POST /search` | POST | Semantic paper search |
| `POST /classify` | POST | Category prediction |
| `POST /summarize` | POST | Text summarisation |
| `POST /ask` | POST | RAG Q&A |
| `POST /agent/run` | POST | Multi-mode agent |
| `POST /agent/run/stream` | POST | Streaming SSE response |
| `POST /chat/upload` | POST | Upload PDF for chat |
| `POST /chat/load-arxiv` | POST | Load arXiv paper by ID |
| `POST /chat/ask` | POST | Chat with loaded paper |
| `GET /chat/session/{id}` | GET | Session metadata |

Full interactive docs at **http://localhost:8000/docs**

---

## 🏗 Project Structure

```
├── frontend/
│   ├── index.html          # Main UI
│   ├── styles.css          # Design system
│   └── app.js              # All client logic
├── src/
│   ├── api/
│   │   ├── main.py         # FastAPI app + routes
│   │   └── schemas.py      # Pydantic models
│   └── research_assistant/
│       ├── rag.py          # RAG retrieval + answer
│       ├── agents.py       # Mode dispatcher
│       ├── paper_chat.py   # PDF/arXiv chat service
│       ├── summarization.py# Summariser (cloud/local)
│       ├── cloud_llm.py    # Unified LLM client
│       ├── similarity.py   # FAISS index utils
│       ├── preprocess.py   # Text cleaning
│       └── config.py       # Config dataclasses
├── pipeline_scripts/       # Training & indexing scripts
├── artifacts/              # Generated model artifacts
│   ├── classification/     # classifier.joblib, tfidf_vectorizer.joblib
│   └── similarity/         # paper_index.faiss, paper_metadata.parquet
├── .env.example
├── requirements.txt
└── start.sh
```

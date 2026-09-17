# Hugging Face Microservice

## File Paths: `hf_microservice/app.py`, `hf_microservice/requirements.txt`
## Status: Active / Remote (Deployed on Hugging Face Spaces)

## Description
This microservice runs completely separately from the Render backend. It is deployed to a ZeroGPU Hugging Face Space to leverage GPU acceleration for embedding models (BGE) and semantic search over the 8,000+ paper FAISS index.

## File Breakdown

### 1. `hf_microservice/app.py`
- **Purpose**: The main entry point for the remote Gradio application.
- **Functions**:
  - `load_data()`: Loads `arxiv_metadata_8k.parquet` and `faiss_index.bin` into RAM on startup.
  - `api_hybrid_search(query, top_k, candidate_k)`: Encodes the query into embeddings using `BAAI/bge-large-en-v1.5`, queries the FAISS index, retrieves metadata, and returns a JSON string containing the results.
  - `api_classify(title, abstract)`: Runs a zero-shot classification model (e.g., `facebook/bart-large-mnli`) to categorize the paper.
  - `api_extract_methodology(text)`: Extracts the research methodology from abstracts using a QA or instruction-tuned model.
  - `api_cluster(papers_json)`: Performs basic K-Means clustering on the retrieved paper embeddings to group them by topic.
- **Endpoints**: Exposed automatically via Gradio `gr.Interface` with `api_name` (e.g., `/hybrid_search_api`).

### 2. `hf_microservice/requirements.txt`
- **Purpose**: Defines dependencies for the HF Space container.
- **Key Packages**:
  - `gradio`, `fastapi`, `uvicorn` (Server framework)
  - `sentence-transformers`, `transformers`, `torch` (ML models)
  - `faiss-cpu` (Vector search)
  - `pandas`, `pyarrow` (Data loading)

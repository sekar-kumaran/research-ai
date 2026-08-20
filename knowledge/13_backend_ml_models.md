# Backend: ML Models

## File Paths: `src/research_ai/ml_models/`
## Status: Active / Legacy

## Description
This directory historically contained local implementations of machine learning models. In v3.1, heavy ML execution has been ENTIRELY offloaded to the remote Hugging Face microservice via `proxy_services.py`. 

**CRITICAL NOTE**: The local classes (`SimilarityService`, `FaissVectorStore`, `ClassifierService`) are now largely legacy/dead code for execution. The `platform.py` explicitly delegates these responsibilities to `RemoteMLClient` and Gradio Space endpoints. They remain in the codebase primarily for interface definition or local-only CI testing, but are NEVER used in production execution.

## File Breakdown

### 1. `src/research_ai/ml_models/classifier/service.py`
- **Class `ClassifierService`**: (Legacy) Local implementation for text classification.
- **Functions**:
  - `classify(title, abstract)`: Historically ran a local zero-shot classifier. Now superseded by `RemoteClassifierService` which hits the microservice `/classify` endpoint.

### 2. `src/research_ai/ml_models/methodology_extractor/service.py`
- **Class `MethodologyExtractor`**: Local fallback for extracting research methods.
- **Functions**:
  - `extract(text)`: Scans abstracts using NLP heuristics or a local instruction-tuned model to identify the study design (e.g., "Randomized Control Trial", "Dataset creation").

### 3. `src/research_ai/ml_models/ranking/service.py`
- **Class `RankingService`**: Re-ranks search results.
- **Functions**:
  - `rerank(query, documents)`: Given a list of initial FAISS results, uses a cross-encoder model to re-score and sort the papers for higher precision.

### 4. `src/research_ai/ml_models/similarity/service.py`
- **Class `SimilarityService`**: Computes text similarity.
- **Functions**:
  - `calculate_similarity(text1, text2)`: Computes cosine similarity between embeddings of two texts to find duplicates or highly related papers.

### 5. `src/research_ai/ml_models/summarizer/service.py`
- **Class `SummarizerService`**:
- **Functions**:
  - `summarize(text)`: Distills long abstracts or full papers into a single paragraph using a local seq2seq model (like BART or T5).

### 6. `src/research_ai/ml_models/citation_graph/service.py`
- **Class `CitationGraph`**:
- **Functions**:
  - `related_signals(papers)`: Analyzes citation networks to find influential foundational papers or calculate a paper's impact factor.

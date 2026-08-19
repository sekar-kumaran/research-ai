# Backend: ML Models

## File Paths: `src/research_ai/ml_models/`
## Status: Active / Legacy

## Description
This directory historically contained local implementations of machine learning models. In v3.1, heavy ML execution has been offloaded to the Hugging Face microservice via `proxy_services.py`. These files remain as interfaces or lightweight local fallbacks.

## File Breakdown

### 1. `src/research_ai/ml_models/classifier/service.py`
- **Class `ClassifierService`**: Local fallback for text classification.
- **Functions**:
  - `classify(title, abstract)`: If the remote HF space is down, this can run a lightweight local zero-shot classifier (if PyTorch is installed locally) to categorize a paper's subject area.

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

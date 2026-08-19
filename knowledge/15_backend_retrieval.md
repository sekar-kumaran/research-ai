# Backend: Retrieval

## File Paths: `src/research_ai/retrieval/`
## Status: Active / Stable

## Description
This module handles Document ingestion, chunking, embedding generation, and vector store querying. For the remote architecture, much of this logic is offloaded to the HF space, but these files provide local capabilities for uploaded PDFs.

## File Breakdown

### 1. `src/research_ai/retrieval/chunking.py`
- **Functions**:
  - `chunk_text(text, chunk_size, overlap)`: Splits long documents into overlapping segments (e.g., 512 tokens) to ensure they fit inside the context window of the embedding model without losing semantic meaning across chunk boundaries.

### 2. `src/research_ai/retrieval/embeddings/service.py`
- **Class `EmbeddingService`**:
- **Functions**:
  - `embed(text)`: Converts text chunks into dense floating-point vectors (embeddings) using models like BGE-large or SentenceTransformers.

### 3. `src/research_ai/retrieval/hybrid_search/service.py`
- **Class `HybridSearch`**:
- **Functions**:
  - `search(query)`: Combines dense vector search (FAISS) with sparse keyword search (BM25) and merges the results using Reciprocal Rank Fusion (RRF) to provide highly accurate retrieval.

### 4. `src/research_ai/retrieval/rerankers/service.py`
- **Class `Reranker`**:
- **Functions**:
  - `rerank()`: A dedicated pipeline step to re-score the top 50 results from hybrid search down to the top 5 most relevant chunks using a cross-encoder.

### 5. `src/research_ai/retrieval/vector_store/faiss_store.py`
- **Class `FaissStore`**:
- **Functions**:
  - `add_vectors(embeddings)`: Inserts vectors into a local FAISS index. Used heavily when users upload custom PDFs that aren't in the remote 8k arXiv database.
  - `search(query_vector)`: Performs L2 or Cosine Similarity nearest-neighbor search.

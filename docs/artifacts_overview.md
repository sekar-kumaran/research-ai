# Artifacts Overview — Research AI Intelligence Platform v3.1

## Expected Artifact Layout

All trained artifacts live under the `ARTIFACTS_ROOT` directory (default: `artifacts/`).

```
artifacts/
├── classification/
│   ├── classifier.joblib          ← Trained sklearn classifier (LogReg / SVM / RF)
│   └── tfidf_vectorizer.joblib    ← TF-IDF vectorizer fitted on arXiv titles+abstracts
└── similarity/
    ├── paper_index.faiss          ← FAISS IndexFlatIP of all paper embeddings
    ├── paper_metadata.parquet     ← Parquet table: id, title, abstract, authors, categories, update_date
    └── embedding_model_name.joblib ← Name of the SentenceTransformer used to build the index
```

## Generating Artifacts

Artifacts are produced by the training pipeline scripts (not included in this release).
The platform degrades gracefully when artifacts are absent:

| Service | Behaviour when artifact missing |
|---|---|
| ClassifierService | Returns `{"error": "Classifier not ready..."}` on every call |
| FaissVectorStore | Returns `{"error": "Search index not ready..."}` on search |
| EmbeddingService | Loads model from HuggingFace Hub on first use (always available) |
| All other services | Fully operational (no artifact dependency) |

## Runtime Loading

All artifact loading is **lazy** — files are opened on the first actual call,
not at server startup. This means:

- Server boot is fast regardless of artifact size.
- The `/health` endpoint reports `classifier: false` / `hybrid_retrieval: false`
  when artifacts are absent, but the server is fully functional for all
  non-artifact-dependent operations.

## Data Root

Raw arXiv data shards (parquet) are expected at `DATA_ROOT/arxiv_chunks/*.parquet`.
The data root is read-only at runtime; the training pipeline writes artifacts.

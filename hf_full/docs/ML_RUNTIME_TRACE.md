# ML Runtime Trace

This document details the exact path and usage of ML artifacts within the Research AI application.

## CLASSIFIER (`ClassifierService`)
- **Artifacts:** `classifier.joblib` (SGDClassifier), `tfidf_vectorizer.joblib` (TfidfVectorizer), `labels.joblib` (List mapping).
- **Loader:** Uses `joblib.load()` internally in `ClassifierService.from_artifacts()`.
- **Consumer:** `/classify` route, `_classify_query` tool, orchestrator intent classification.
- **Input:** Text strings (title + abstract).
- **Preprocessing:** Standard lowercasing/regex stripping (via `TfidfVectorizer`).
- **Prediction Path:** Converts text to TF-IDF vector, passes to `SGDClassifier.predict()`, decodes integer to string via `labels.joblib`.
- **Device:** CPU (Scikit-learn).
- **Load Timing:** Eagerly instantiated by `_prewarm()`.

## EMBEDDING (`EmbeddingService`)
- **Artifacts:** `embedding_model_name.joblib` (or loaded directly via `SentenceTransformer("all-MiniLM-L6-v2")`).
- **Loader:** Uses `SentenceTransformer()` in `EmbeddingService.__init__()`.
- **Consumer:** `HybridSearchService`, `/search`, `/similarity`, `SimilarityService`.
- **Input:** Query string or list of text chunks.
- **Output:** Dense Numpy vectors (384 dimensions).
- **Normalization:** `normalize_embeddings=True` used implicitly for FAISS Inner Product metric.
- **Device:** CPU by default. Needs `@spaces.GPU` applied dynamically for ZeroGPU compatibility.
- **Load Timing:** Eagerly instantiated by `_prewarm()`.

## FAISS (`FaissVectorStore`)
- **Artifacts:** `paper_index.faiss`, `paper_metadata.parquet`.
- **Loader:** `faiss.read_index()` for the index, `pd.read_parquet()` for metadata. Loaded lazily on first `.search()`.
- **Consumer:** `HybridSearchService`, `/search`.
- **Index Type:** Typically `IndexFlatIP` (Inner Product).
- **Dimensionality:** 384.
- **Metadata Lookup:** FAISS returns index IDs, which map to DataFrame row indices to extract `title`, `abstract`, and `year`.
- **Device:** CPU (`faiss-cpu`).
- **Load Timing:** Lazily loaded conceptually, but immediately forced into RAM by `_prewarm()`.

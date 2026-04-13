from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

try:
    import faiss
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("faiss-cpu is required for similarity index.") from exc


def build_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    return SentenceTransformer(model_name)


def encode_texts(model: SentenceTransformer, texts: list[str], batch_size: int = 128) -> np.ndarray:
    vectors = model.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)
    # L2 normalize so inner product becomes cosine similarity.
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.clip(norms, 1e-12, None)


def build_faiss_index(vectors: np.ndarray):
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors.astype(np.float32))
    return index


def search_similar(index, query_vec: np.ndarray, top_k: int = 5):
    scores, ids = index.search(query_vec.astype(np.float32), top_k)
    return scores[0], ids[0]


def save_similarity_artifacts(artifact_dir, index, metadata_df: pd.DataFrame):
    artifact_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(artifact_dir / "paper_index.faiss"))
    metadata_df.to_parquet(artifact_dir / "paper_metadata.parquet", index=False)


def load_similarity_artifacts(artifact_dir):
    index = faiss.read_index(str(artifact_dir / "paper_index.faiss"))
    metadata = pd.read_parquet(artifact_dir / "paper_metadata.parquet")
    return index, metadata


def save_embedding_model_name(artifact_dir, model_name: str):
    artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_name, artifact_dir / "embedding_model_name.joblib")

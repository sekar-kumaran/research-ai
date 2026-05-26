"""FAISS vector store wrapper — semantic paper retrieval over indexed embeddings.

DATA MODEL
----------
Two artifacts form the store:
  paper_index.faiss     — FAISS IndexFlatIP (inner product, i.e. cosine for
                          L2-normalised vectors). One vector per paper.
  paper_metadata.parquet— One row per paper, positionally aligned to the FAISS
                          index.  Row i ↔ vector i.

POSITIONAL ALIGNMENT INVARIANT
--------------------------------
The FAISS index stores raw integer row indices (0, 1, 2, …).  search() maps
these back to metadata rows via `metadata.iloc[idx]`.  This means the
metadata DataFrame MUST be ordered identically to the embedding order used
when building the index.  Never sort, filter, or reset_index on the metadata
after index construction without rebuilding the FAISS index.

LAZY LOADING
------------
Artifacts are not read from disk until the first search() call.  This keeps
startup fast and allows the web server to accept health-check requests before
the 1–2 GB index is fully loaded into RAM.

DIMENSION VALIDATION (BUG FIX v3.1.1)
---------------------------------------
Previously there was no check that the query vector dimension matched the
FAISS index dimension.  If an administrator changed EMBEDDING_MODEL in .env
after building the index, all searches would silently return wrong results or
crash with an unintelligible FAISS error.

Fix: _ensure_loaded() now reads index.d (the stored embedding dimension) and
raises a clear RuntimeError if a query vector of a different dimension is seen
in search().  This turns a silent data-corruption bug into a fail-fast error.

INDEX TYPE CHOICE
-----------------
IndexFlatIP is an exact (non-approximate) brute-force inner-product search.
- Pros: 100% recall, no training required, deterministic results.
- Cons: O(N × d) per query — linear in corpus size.
- For 100k papers × 384 dims, this is ~40M multiplications, typically <5ms
  with FAISS's BLAS-backed implementation on a modern CPU.
- If the corpus grows to >500k papers, consider IndexHNSWFlat (approximate,
  sub-linear) or IndexIVFFlat with nlist~100 clusters.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

try:
    import faiss
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("faiss-cpu is required for vector retrieval.") from exc

logger = logging.getLogger(__name__)


@dataclass
class RetrievedDocument:
    """A single paper retrieved from the FAISS index.

    score: inner-product similarity ∈ [0, 1] for L2-normalised vectors.
           Higher is more similar.  Equivalent to cosine similarity.
    year:  extracted from the first 4 chars of update_date (arXiv convention).
    """
    paper_id: str
    title: str
    abstract: str
    score: float
    authors: str = ""
    category: str = ""
    year: str = ""

    def to_dict(self) -> dict:
        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "abstract": self.abstract,
            "score": round(self.score, 4),
            "authors": self.authors,
            "category": self.category,
            "year": self.year,
        }


class FaissVectorStore:
    """Wraps a FAISS flat index + aligned Parquet metadata for paper retrieval.

    Typical usage:
        store = FaissVectorStore.from_artifacts(Path("artifacts/similarity"))
        results = store.search(query_vec, top_k=10)

    The store is safe to share across threads — FAISS IndexFlatIP is read-only
    after loading and pandas DataFrames are immutable for .iloc access.
    """

    def __init__(
        self,
        index=None,
        metadata: pd.DataFrame | None = None,
        artifact_dir: Path | None = None,
    ) -> None:
        self.index = index
        self.metadata = metadata if metadata is not None else pd.DataFrame()
        self.artifact_dir = artifact_dir

    @classmethod
    def from_artifacts(cls, artifact_dir: Path) -> "FaissVectorStore":
        """Create a lazy store backed by artifact files (loaded on first search)."""
        return cls(artifact_dir=artifact_dir)

    @property
    def ready(self) -> bool:
        """True if the store is either already loaded or its artifact files exist."""
        if self.index is not None and not self.metadata.empty:
            return True
        if self.artifact_dir is None:
            return False
        return (
            (self.artifact_dir / "paper_index.faiss").exists()
            and (self.artifact_dir / "paper_metadata.parquet").exists()
        )

    @property
    def paper_count(self) -> int:
        """Number of indexed papers.  Reads Parquet footer only (no data load)."""
        if not self.metadata.empty:
            return int(len(self.metadata))
        if self.artifact_dir is None:
            return 0
        metadata_path = self.artifact_dir / "paper_metadata.parquet"
        if not metadata_path.exists():
            return 0
        try:
            # pq.ParquetFile.metadata.num_rows reads only the file footer —
            # extremely fast even for multi-GB Parquet files.
            return int(pq.ParquetFile(metadata_path).metadata.num_rows)
        except Exception:
            return 0

    def _ensure_loaded(self) -> None:
        """Load FAISS index and metadata from disk (idempotent after first call).

        Raises RuntimeError if artifacts are missing or corrupted.
        """
        if self.index is not None and not self.metadata.empty:
            return  # already loaded
        if self.artifact_dir is None:
            raise RuntimeError("Vector store artifact directory is not configured.")
        index_path = self.artifact_dir / "paper_index.faiss"
        metadata_path = self.artifact_dir / "paper_metadata.parquet"
        if not index_path.exists() or not metadata_path.exists():
            raise RuntimeError(
                f"Vector store artifacts are missing in {self.artifact_dir}. "
                "Run the embedding/indexing pipeline first."
            )
        self.index = faiss.read_index(str(index_path))
        self.metadata = pd.read_parquet(metadata_path)

        # Validate positional alignment: FAISS ntotal must equal metadata rows.
        # A mismatch means the index and metadata were built from different data.
        if self.index.ntotal != len(self.metadata):
            raise RuntimeError(
                f"FAISS index has {self.index.ntotal} vectors but metadata has "
                f"{len(self.metadata)} rows.  Rebuild the similarity artifacts."
            )
        logger.info(
            "FaissVectorStore loaded: %d papers, dim=%d",
            self.index.ntotal, self.index.d,
        )

    def search(self, query_vec: np.ndarray, top_k: int) -> list[RetrievedDocument]:
        """Search for the top_k most similar papers to query_vec.

        Args:
            query_vec: L2-normalised query embedding, shape (1, d) or (d,).
                       MUST match the dimension used when building the index.
            top_k:     Number of nearest neighbours to return.

        Returns:
            List of RetrievedDocument, sorted by score descending.
            Empty list if the store is not ready or all FAISS IDs are -1.

        Raises:
            RuntimeError: if query_vec dimension does not match index dimension.
                          This is a configuration error — rebuild with the
                          correct embedding model.
        """
        if not self.ready:
            return []
        self._ensure_loaded()

        # DIMENSION VALIDATION (BUG FIX v3.1.1):
        # FAISS stores the embedding dimension as index.d.  If the embedding
        # model was changed after index construction, query_vec.shape[-1] will
        # differ from index.d, causing silent wrong results or a cryptic FAISS
        # error.  Detect this early and give a clear actionable error message.
        query_dim = query_vec.shape[-1]
        if query_dim != self.index.d:
            raise RuntimeError(
                f"Embedding dimension mismatch: query vector has {query_dim} dims "
                f"but FAISS index was built with {self.index.d} dims.  "
                "Change EMBEDDING_MODEL back to the model used during indexing, "
                "or rebuild the FAISS index with the new model."
            )

        scores, ids = self.index.search(query_vec.astype("float32"), top_k)
        docs: list[RetrievedDocument] = []
        for score, idx in zip(scores[0], ids[0]):
            # FAISS returns -1 for "no result" when fewer than top_k docs exist
            if idx < 0 or idx >= len(self.metadata):
                continue
            row = self.metadata.iloc[int(idx)]
            docs.append(
                RetrievedDocument(
                    paper_id=str(row.get("id", "")),
                    title=str(row.get("title", "Untitled")).strip(),
                    abstract=str(row.get("abstract", "")).strip(),
                    score=float(score),
                    authors=str(row.get("authors", "")),
                    category=str(row.get("categories", "")),
                    year=str(row.get("update_date", ""))[:4],
                )
            )
        return docs

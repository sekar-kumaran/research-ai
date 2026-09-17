from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ArtifactCheck:
    name: str
    path: str
    ready: bool
    detail: str


class ArtifactManager:
    """Validate local ML artifacts and report feature readiness."""

    LFS_PREFIX = b"version https://git-lfs.github.com"

    def __init__(self, root: Path) -> None:
        self.root = root

    def report(self) -> dict:
        checks = [
            self._joblib("classifier", self.root / "classification" / "classifier.joblib"),
            self._joblib("tfidf_vectorizer", self.root / "classification" / "tfidf_vectorizer.joblib"),
            self._faiss("paper_index", self.root / "similarity" / "paper_index.faiss"),
            self._parquet("paper_metadata", self.root / "similarity" / "paper_metadata.parquet"),
            self._joblib("embedding_model_name", self.root / "similarity" / "embedding_model_name.joblib"),
            self._joblib("kmeans", self.root / "clustering" / "kmeans.joblib"),
            self._parquet("cluster_assignments", self.root / "clustering" / "cluster_assignments.parquet"),
        ]
        return {
            "root": str(self.root),
            "checks": [check.__dict__ for check in checks],
            "classification_ready": all(c.ready for c in checks[:2]),
            "faiss_ready": checks[2].ready and checks[3].ready,
            "clustering_ready": checks[5].ready and checks[6].ready,
        }

    def log_report(self) -> None:
        report = self.report()
        lines = ["ML ARTIFACT STATUS"]
        for item in report["checks"]:
            marker = "READY" if item["ready"] else "UNAVAILABLE"
            lines.append(f"{marker:12} {item['name']}: {item['detail']}")
        logger.info("\n".join(lines))

    def _base(self, name: str, path: Path, min_bytes: int = 1) -> ArtifactCheck | None:
        if not path.exists():
            return ArtifactCheck(name, str(path), False, "missing")
        size = path.stat().st_size
        if size < min_bytes:
            return ArtifactCheck(name, str(path), False, f"too small ({size} bytes)")
        try:
            with path.open("rb") as fh:
                if fh.read(len(self.LFS_PREFIX)).startswith(self.LFS_PREFIX):
                    return ArtifactCheck(name, str(path), False, "git-lfs pointer, not a real artifact")
        except OSError as exc:
            return ArtifactCheck(name, str(path), False, str(exc))
        return None

    def _joblib(self, name: str, path: Path) -> ArtifactCheck:
        base = self._base(name, path, min_bytes=16)
        if base:
            return base
        try:
            joblib.load(path)
            return ArtifactCheck(name, str(path), True, "load ok")
        except Exception as exc:
            return ArtifactCheck(name, str(path), False, f"joblib load failed: {exc}")

    def _parquet(self, name: str, path: Path) -> ArtifactCheck:
        base = self._base(name, path, min_bytes=512)
        if base:
            return base
        try:
            rows = len(pd.read_parquet(path))
            return ArtifactCheck(name, str(path), rows > 0, f"{rows} rows")
        except Exception as exc:
            return ArtifactCheck(name, str(path), False, f"parquet load failed: {exc}")

    def _faiss(self, name: str, path: Path) -> ArtifactCheck:
        base = self._base(name, path, min_bytes=1024)
        if base:
            return base
        try:
            import faiss

            index = faiss.read_index(str(path))
            return ArtifactCheck(name, str(path), index.ntotal > 0, f"{index.ntotal} vectors, dim={index.d}")
        except Exception as exc:
            return ArtifactCheck(name, str(path), False, f"faiss load failed: {exc}")

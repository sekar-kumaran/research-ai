"""Download large ResearchAI artifacts from a Hugging Face Dataset repo.

The application starts without optional artifacts, but features that require
them report unavailable until the real files are present. This script fills the
canonical artifact directory configured by ARTIFACTS_ROOT.
"""
from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

ARTIFACTS = [
    ("artifacts/classification/classifier.joblib", "classification/classifier.joblib", "~608 MB"),
    ("artifacts/clustering/kmeans.joblib", "clustering/kmeans.joblib", "~80 MB"),
    ("artifacts/clustering/cluster_assignments.parquet", "clustering/cluster_assignments.parquet", "~7 MB"),
    ("artifacts/similarity/paper_index.faiss", "similarity/paper_index.faiss", "~12 MB"),
    ("artifacts/similarity/paper_metadata.parquet", "similarity/paper_metadata.parquet", "~5 MB"),
    ("artifacts/similarity/embedding_model_name.joblib", "similarity/embedding_model_name.joblib", "~1 KB"),
]

LFS_STUB_PREFIX = b"version https://git-lfs.github.com"


def _is_lfs_stub(path: Path) -> bool:
    try:
        with path.open("rb") as fh:
            return fh.read(len(LFS_STUB_PREFIX)).startswith(LFS_STUB_PREFIX)
    except OSError:
        return False


def _is_valid_existing(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0 and not _is_lfs_stub(path)


def download_artifacts(repo_id: str) -> None:
    """Download missing artifacts into ARTIFACTS_ROOT.

    The HF dataset is expected to contain files under an ``artifacts/`` prefix,
    while the local runtime directory contains the inner folders directly:
    ``classification/``, ``similarity/``, and ``clustering/``.
    """
    if not repo_id:
        logger.info("HF_ARTIFACTS_REPO not set - skipping artifact download.")
        return

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        logger.warning("huggingface_hub is not installed - cannot download artifacts.")
        return

    token = os.getenv("HF_TOKEN", "").strip() or None
    artifact_root = Path(os.getenv("ARTIFACTS_ROOT", "artifacts"))
    artifact_root.mkdir(parents=True, exist_ok=True)

    downloaded_count = 0
    skipped_count = 0

    for repo_path, artifact_rel_path, size_hint in ARTIFACTS:
        local_path = artifact_root / artifact_rel_path

        if _is_valid_existing(local_path):
            logger.info("Artifact already present - skipping: %s", local_path)
            skipped_count += 1
            continue

        if local_path.exists() and _is_lfs_stub(local_path):
            logger.warning("Git-LFS pointer stub detected at %s - replacing it.", local_path)
            local_path.unlink()

        local_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading %s (%s) from %s ...", repo_path, size_hint, repo_id)

        try:
            downloaded = hf_hub_download(
                repo_id=repo_id,
                filename=repo_path,
                repo_type="dataset",
                token=token,
            )
            shutil.copyfile(downloaded, local_path)
            downloaded_count += 1
            logger.info("Downloaded: %s -> %s", repo_path, local_path)
        except Exception as exc:
            logger.warning(
                "Could not download %s from %s: %s. Feature remains unavailable.",
                repo_path,
                repo_id,
                exc,
            )

    logger.info(
        "Artifact download complete: %d downloaded, %d already present.",
        downloaded_count,
        skipped_count,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    download_artifacts(os.getenv("HF_ARTIFACTS_REPO", "").strip())

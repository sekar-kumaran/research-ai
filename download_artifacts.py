"""download_artifacts.py — Downloads large precomputed artifacts from a Hugging Face Dataset repo.

WHY THIS EXISTS
---------------
Some artifacts are too large to commit directly to the HF Space repository:
  - artifacts/classification/classifier.joblib  (~608 MB)
  - artifacts/clustering/kmeans.joblib           (~80 MB)

These are precomputed once and stored in a separate HF Dataset repo. This
script downloads them at Space startup *only if they are missing locally*.
Subsequent startups skip the download entirely (cached on disk).

USAGE
-----
Set the HF_ARTIFACTS_REPO environment variable to your dataset repo:

    HF_ARTIFACTS_REPO=sekar-kumaran/research-ai-artifacts

The repo should have the same directory layout as the local artifacts/ dir:
    artifacts/classification/classifier.joblib
    artifacts/clustering/kmeans.joblib
    etc.

If HF_ARTIFACTS_REPO is not set, this script does nothing and the app starts
normally. Missing artifacts only cause errors when those specific features
are first used (classifier, clustering).

AUTHENTICATION
--------------
For public HF Dataset repos, no token is needed.
For private repos, set HF_TOKEN as a Hugging Face Space Secret.

FILES DOWNLOADED
----------------
Only files that are MISSING locally are downloaded. If the file already
exists, the download is skipped (idempotent startup).
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# Artifacts required at runtime and their sizes (for logging).
# Files tracked by Git-LFS that must be downloaded from HF Dataset repo instead.
# This includes both "too large to commit" files AND files that GitHub's LFS
# bandwidth limits caused to become pointer stubs in the deployed container.
_LARGE_ARTIFACTS = [
    # (path relative to repo root, approx size for logging)
    ("artifacts/classification/classifier.joblib", "~608 MB"),
    ("artifacts/clustering/kmeans.joblib", "~80 MB"),
    ("artifacts/clustering/cluster_assignments.parquet", "~7 MB"),
    # FAISS similarity index and metadata \u2014 tracked by Git-LFS but frequently
    # become un-resolved stubs on GitHub due to LFS bandwidth/storage limits.
    # Must be downloaded from HF Dataset repo instead.
    ("artifacts/similarity/paper_index.faiss", "~12 MB"),
    ("artifacts/similarity/paper_metadata.parquet", "~5 MB"),
]

# Git-LFS pointer stub prefix \u2014 files starting with this are NOT real artifacts.
_LFS_STUB_PREFIX = b"version https://git-lfs.github.com"


def _is_lfs_stub(path: Path) -> bool:
    """Return True if a file is an unresolved Git-LFS pointer stub."""
    try:
        with path.open("rb") as fh:
            return fh.read(36).startswith(_LFS_STUB_PREFIX)
    except OSError:
        return False



def download_artifacts(repo_id: str) -> None:
    """Download missing large artifacts from a HF Dataset repo.

    Args:
        repo_id: HF Dataset repo ID, e.g. "sekar-kumaran/research-ai-artifacts"
    """
    if not repo_id:
        logger.info("HF_ARTIFACTS_REPO not set — skipping artifact download.")
        return

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        logger.warning(
            "huggingface_hub is not installed — cannot download artifacts. "
            "Install it with: pip install huggingface_hub"
        )
        return

    token = os.getenv("HF_TOKEN", "").strip() or None
    repo_root = Path(__file__).resolve().parent

    downloaded_count = 0
    skipped_count = 0

    for rel_path, size_hint in _LARGE_ARTIFACTS:
        local_path = repo_root / rel_path

        if local_path.exists():
            if _is_lfs_stub(local_path):
                logger.warning(
                    "Artifact at %s is a Git-LFS pointer stub (not real data). "
                    "Re-downloading from %s ...", rel_path, repo_id
                )
                local_path.unlink()  # remove stub so download proceeds
            else:
                logger.info("Artifact already present — skipping: %s", rel_path)
                skipped_count += 1
                continue


        # Ensure parent directory exists
        local_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Downloading %s (%s) from %s ...", rel_path, size_hint, repo_id
        )
        try:
            downloaded = hf_hub_download(
                repo_id=repo_id,
                filename=rel_path,
                repo_type="dataset",
                token=token,
                local_dir=str(repo_root),
                local_dir_use_symlinks=False,
            )
            logger.info("Downloaded: %s → %s", rel_path, downloaded)
            downloaded_count += 1
        except Exception as exc:
            # Non-fatal: the app starts without this artifact.
            # Features that need it (classify, cluster) will fail gracefully
            # with a clear error at request time.
            logger.warning(
                "Could not download %s from %s: %s. "
                "Features requiring this artifact will be unavailable.",
                rel_path, repo_id, exc,
            )

    logger.info(
        "Artifact download complete: %d downloaded, %d already present.",
        downloaded_count, skipped_count,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    _repo = os.getenv("HF_ARTIFACTS_REPO", "").strip()
    if not _repo:
        print("Usage: HF_ARTIFACTS_REPO=owner/repo python download_artifacts.py")
    else:
        download_artifacts(_repo)

import subprocess
from pathlib import Path
import pytest

def test_no_tracked_pycache():
    """Ensure no __pycache__ or .pyc files are tracked in git."""
    result = subprocess.run(
        ["git", "ls-files"], 
        capture_output=True, 
        text=True, 
        check=True
    )
    tracked_files = result.stdout.splitlines()
    
    bad_files = [f for f in tracked_files if "__pycache__" in f or f.endswith(".pyc")]
    
    assert not bad_files, f"Found tracked pycache files: {bad_files}. Run 'git rm --cached' on them."

def test_no_lfs_stubs_in_prod_artifacts():
    """Ensure that any *.faiss or *.parquet files under artifacts/ are real files, not LFS stubs.
    
    This guards against silent deployment failures where Git-LFS fails to resolve.
    """
    artifacts_dir = Path("artifacts")
    if not artifacts_dir.exists():
        return
        
    _LFS_STUB_PREFIX = b"version https://git-lfs.github.com"
    
    for ext in ["*.faiss", "*.parquet", "*.joblib"]:
        for path in artifacts_dir.rglob(ext):
            # Read first few bytes to check for LFS stub
            with path.open("rb") as f:
                header = f.read(36)
            
            assert not header.startswith(_LFS_STUB_PREFIX), (
                f"File {path} is an unresolved Git-LFS pointer stub! "
                f"Ensure it is either downloaded via HF Datasets or properly resolved via git-lfs."
            )
            
            # Additional sanity check on file size for FAISS
            if path.suffix == ".faiss":
                assert path.stat().st_size > 1024, f"FAISS index {path} is suspiciously small (< 1KB)."

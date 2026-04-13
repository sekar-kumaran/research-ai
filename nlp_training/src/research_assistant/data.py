from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = ["id", "title", "abstract", "categories"]


def load_arxiv_frame(shards: list[Path], use_columns: list[str] | None = None) -> pd.DataFrame:
    """Load and concatenate parquet shards.

    Parameters:
    - shards: list of parquet files.
    - use_columns: optional columns projection for faster IO.
    """
    if not shards:
        raise FileNotFoundError("No parquet shards found. Check dataset/arxiv_chunks.")

    cols = use_columns or REQUIRED_COLUMNS
    frames: list[pd.DataFrame] = []

    for shard in shards:
        frames.append(pd.read_parquet(shard, columns=cols))

    return pd.concat(frames, ignore_index=True)


def add_category_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Create normalized category targets from the raw categories field."""
    out = df.copy()
    out["primary_category"] = out["categories"].astype(str).str.split().str[0]
    out["broad_category"] = out["primary_category"].astype(str).str.split(".").str[0]
    return out


def keep_balanced_domains(df: pd.DataFrame, domains: tuple[str, ...], sample_per_class: int, random_state: int) -> pd.DataFrame:
    """Filter target domains and stratified-sample equal rows per class."""
    filtered = df[df["broad_category"].isin(domains)].copy()

    # Avoid pandas groupby.apply semantics changes across versions by sampling
    # each class explicitly and concatenating the results.
    parts: list[pd.DataFrame] = []
    for domain in domains:
        group = filtered[filtered["broad_category"] == domain]
        if group.empty:
            continue
        n = min(len(group), sample_per_class)
        parts.append(group.sample(n=n, random_state=random_state))

    if not parts:
        raise ValueError("No rows found for requested target domains.")

    sampled = pd.concat(parts, ignore_index=True)
    return sampled

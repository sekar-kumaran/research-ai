from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    data_root: Path
    artifacts_root: Path

    @property
    def arxiv_shards(self) -> list[Path]:
        return sorted((self.data_root / "arxiv_chunks").glob("*.parquet"))


@dataclass(frozen=True)
class TextConfig:
    # Remove very short tokens that usually add noise.
    min_token_len: int = 3
    # Limit vocabulary size for scalable TF-IDF training.
    max_features: int = 30000
    # Keep unigrams and bigrams.
    ngram_range: tuple[int, int] = (1, 2)
    # Ignore terms that appear in too many documents.
    max_df: float = 0.9
    # Ignore terms that are too rare.
    min_df: int = 5


@dataclass(frozen=True)
class TrainConfig:
    random_state: int = 42
    test_size: float = 0.2
    sample_per_class: int = 20000
    target_domains: tuple[str, ...] = ("cs", "math", "physics", "q-bio")

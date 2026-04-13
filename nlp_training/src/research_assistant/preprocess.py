from __future__ import annotations

import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download once; if already present these calls are no-op.
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

_STOPWORDS = set(stopwords.words("english"))
_STOPWORDS.update({"paper", "study", "approach", "method", "results", "using", "show", "propose"})
_LEMMATIZER = WordNetLemmatizer()


def build_full_text(title: str, abstract: str) -> str:
    return f"{title or ''} {abstract or ''}".strip()


def clean_text(text: str, min_token_len: int = 3) -> str:
    """Normalize scientific text for classical ML pipelines."""
    value = str(text).lower()
    value = re.sub(r"\$.*?\$", " ", value)
    value = re.sub(r"http\S+|www\.\S+", " ", value)
    value = re.sub(r"\S+@\S+", " ", value)
    value = re.sub(r"[^a-z\s]", " ", value)

    tokens = []
    for token in value.split():
        if token in _STOPWORDS:
            continue
        if len(token) < min_token_len:
            continue
        tokens.append(_LEMMATIZER.lemmatize(token))

    return " ".join(tokens)

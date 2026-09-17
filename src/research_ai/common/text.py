from __future__ import annotations

import re

try:
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer

    _STOPWORDS = set(stopwords.words("english"))
    _LEMMATIZER = WordNetLemmatizer()
except Exception:
    _STOPWORDS = {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
        "in", "is", "it", "its", "of", "on", "or", "that", "the", "this", "to",
        "was", "were", "with",
    }
    _LEMMATIZER = None

_STOPWORDS.update({"paper", "study", "approach", "method", "results", "using", "show", "propose"})


def build_full_text(title: str, abstract: str) -> str:
    return f"{title or ''} {abstract or ''}".strip()


def clean_text(text: str, min_token_len: int = 3) -> str:
    value = str(text).lower()
    value = re.sub(r"\$.*?\$", " ", value)
    value = re.sub(r"http\S+|www\.\S+", " ", value)
    value = re.sub(r"\S+@\S+", " ", value)
    value = re.sub(r"[^a-z\s]", " ", value)
    tokens = []
    for token in value.split():
        if token in _STOPWORDS or len(token) < min_token_len:
            continue
        if _LEMMATIZER is not None:
            try:
                token = _LEMMATIZER.lemmatize(token)
            except Exception:
                pass
        tokens.append(token)
    return " ".join(tokens)


def redact_secrets(text: str) -> str:
    import os

    value = str(text)
    for env_name in ("GROQ_API_KEY", "OPENROUTER_API_KEY", "GOOGLE_API_KEY"):
        secret = os.getenv(env_name, "")
        if secret:
            value = value.replace(secret, "[redacted]")
    return re.sub(r"([?&]key=)[^&\s')]+", r"\1[redacted]", value)


def tokenize_query(text: str) -> set[str]:
    return set(clean_text(text, min_token_len=2).split())

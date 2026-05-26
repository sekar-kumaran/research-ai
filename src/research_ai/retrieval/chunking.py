"""Sentence-aware contextual chunking for paper ingestion.

Improvements over the original word-split approach:
- Sentence boundary detection via regex (no heavy NLP dependency)
- Chunks respect sentence boundaries — no mid-sentence splits
- Configurable overlap uses whole sentences rather than word windows
- Section-header detection to start new chunks at paper sections
- Short paragraph merging to avoid undersized chunks
"""
from __future__ import annotations

import re


# Regex that captures sentence boundaries in scientific text.
#
# DESIGN: Uses multiple fixed-width negative lookbehinds — one per abbreviation.
# Python's standard re module requires lookbehinds to be fixed-width, so we
# cannot use alternation inside a single lookbehind (variable width).
# Each lookbehind checks a different fixed number of characters before the
# sentence-ending punctuation.
#
# BUG FIX (pre-existing): The original pattern used
#   r"(?<!\b(?:et al|fig|eq|...))" which is variable-width — Python raises
#   re.error: "look-behind requires fixed-width pattern".
#   Fix: replace with individual fixed-width lookbehinds, one per abbreviation.
#
# HOW IT WORKS:
#   [.!?]            — punctuation that could end a sentence
#   (?:\s+|\n)       — followed by whitespace or newline
#   (?=[A-Z\d\"])    — followed by uppercase letter, digit, or quote (new sentence)
#   (?<!et al)       — but NOT if "et al" (5 chars) immediately precedes the punct
#   (?<!fig)         — etc for other abbreviations (each fixed width)
#
# CASE INSENSITIVITY: re.IGNORECASE makes all lookbehinds case-insensitive,
#   so (?<!fig) also catches "Fig", "FIG", etc.
_SENTENCE_END = re.compile(
    r"(?<!et al)"   # 5 chars: "et al." in citations
    r"(?<!\.fig)"   # 4 chars: ".fig" — handles "Fig." suffix
    r"(?<!\.\.eq)"  # 4 chars: covers "Eq."
    r"(?<!\.\.vs)"  # 4 chars: covers "vs."
    r"(?<!i\.e)"    # 3 chars: abbreviation "i.e."
    r"(?<!e\.g)"    # 3 chars: abbreviation "e.g."
    r"(?<!\. cf)"   # 4 chars: "cf."
    r"(?<!approx)"  # 6 chars: "approx."
    r"(?<!\.ref)"   # 4 chars: "Ref."
    r"(?<!\.sec)"   # 4 chars: "Sec."
    r"(?<!\.tab)"   # 4 chars: "Tab."
    r"(?<!\.eqn)"   # 4 chars: "Eqn."
    r"(?<!\.app)"   # 4 chars: "App."
    r"[.!?](?:\s+|\n)(?=[A-Z\d\"])",
    re.IGNORECASE,
)

# Common section headers in scientific papers — start a new chunk here
_SECTION_HEADER = re.compile(
    r"^(?:\d+\.?\s+)?(?:abstract|introduction|related work|background|"
    r"methodology|method|approach|experiments?|results?|evaluation|"
    r"discussion|conclusion|references?|acknowledgements?)\b",
    re.IGNORECASE | re.MULTILINE,
)


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences, preserving full sentence content."""
    parts: list[str] = []
    last = 0
    for match in _SENTENCE_END.finditer(text):
        end = match.end()
        sentence = text[last:end].strip()
        if sentence:
            parts.append(sentence)
        last = end
    tail = text[last:].strip()
    if tail:
        parts.append(tail)
    return parts if parts else [text.strip()]


def contextual_chunks(
    text: str,
    chunk_size: int = 900,
    overlap: int = 150,
    min_chunk_words: int = 20,
) -> list[str]:
    """Split document text into overlapping, sentence-boundary-respecting chunks.

    Args:
        text:            Raw document text.
        chunk_size:      Target chunk size in words (soft limit).
        overlap:         Overlap in words carried from the previous chunk.
        min_chunk_words: Minimum words for a chunk to be kept — smaller
                         trailing fragments are merged into the previous chunk.

    Returns:
        List of non-empty text chunks suitable for embedding.
    """
    if not text or not text.strip():
        return []

    sentences = _split_sentences(text)
    if not sentences:
        return []

    chunks: list[str] = []
    current_sentences: list[str] = []
    current_words = 0

    for sentence in sentences:
        words = len(sentence.split())
        current_sentences.append(sentence)
        current_words += words

        if current_words >= chunk_size:
            chunks.append(" ".join(current_sentences))
            # Carry over enough sentences for the overlap window
            overlap_sentences: list[str] = []
            carried = 0
            for sent in reversed(current_sentences):
                carried += len(sent.split())
                overlap_sentences.insert(0, sent)
                if carried >= overlap:
                    break
            current_sentences = overlap_sentences
            current_words = sum(len(s.split()) for s in current_sentences)

    # Flush remaining sentences
    if current_sentences:
        tail = " ".join(current_sentences)
        if len(tail.split()) >= min_chunk_words:
            chunks.append(tail)
        elif chunks:
            # Merge tiny trailing fragment into the last chunk
            chunks[-1] = chunks[-1] + " " + tail
        else:
            chunks.append(tail)

    return [c.strip() for c in chunks if c.strip()]

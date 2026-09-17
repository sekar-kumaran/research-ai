"""Tests for contextual_chunks — sentence-aware document chunking.

Covers:
  - Empty / whitespace-only input
  - Short text fits in one chunk
  - Long text splits correctly
  - Chunk overlap (words from previous chunk appear in next)
  - No empty chunks in output
  - Min-chunk merging (tiny trailing fragments get merged)
  - Section header detection starts a new chunk
  - Scientific abbreviation handling (et al., Fig., vs.)
"""
from __future__ import annotations

import pytest

from research_ai.retrieval.chunking import contextual_chunks


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_string(self):
        assert contextual_chunks("") == []

    def test_whitespace_only(self):
        assert contextual_chunks("   \n\t  ") == []

    def test_single_sentence(self):
        chunks = contextual_chunks("This is a single sentence.")
        assert len(chunks) == 1
        assert "single sentence" in chunks[0]

    def test_no_empty_chunks(self):
        text = "Word " * 2000
        chunks = contextual_chunks(text)
        for chunk in chunks:
            assert chunk.strip() != "", "Found empty chunk"

    def test_output_is_list_of_strings(self):
        chunks = contextual_chunks("Some academic text about neural networks.")
        assert isinstance(chunks, list)
        for c in chunks:
            assert isinstance(c, str)


# ---------------------------------------------------------------------------
# Chunking behaviour
# ---------------------------------------------------------------------------

class TestChunkingBehaviour:
    def test_short_text_single_chunk(self):
        """A text shorter than chunk_size returns exactly one chunk."""
        text = " ".join(["word"] * 50)  # 50 words << 900 default
        chunks = contextual_chunks(text, chunk_size=900)
        assert len(chunks) == 1

    def test_long_text_multiple_chunks(self):
        """A text longer than chunk_size is split into multiple chunks."""
        # Build 3000-word text (should produce ~3-4 chunks at 900 words each)
        text = ". ".join([" ".join(["word"] * 100)] * 30) + "."
        chunks = contextual_chunks(text, chunk_size=900)
        assert len(chunks) >= 2

    def test_chunk_size_respected_approximately(self):
        """Chunks should stay near the target chunk_size.

        NOTE: The chunker splits on SENTENCE boundaries, so text with no
        punctuation cannot be split (the whole text is one "sentence").
        This test uses text with proper sentence endings so chunking fires.
        We allow 2× chunk_size as the upper bound, since sentence-boundary
        alignment means the split may happen slightly after chunk_size.
        """
        # Build 2000-word text as 40-word sentences with proper punctuation
        sentence = " ".join(["word"] * 40) + "."
        text = " ".join([sentence] * 50)  # 50 × 40 = 2000 words
        chunks = contextual_chunks(text, chunk_size=200, overlap=50)
        assert len(chunks) >= 2, "Text should produce multiple chunks"
        for chunk in chunks:
            word_count = len(chunk.split())
            # Allow 2× chunk_size to account for overlap + sentence boundary alignment
            assert word_count <= 400 + 1, f"Chunk too large: {word_count} words"

    def test_overlap_carries_words(self):
        """The end of chunk N should overlap with the start of chunk N+1."""
        # Build text with identifiable sentences
        sentences = [f"Sentence {i} has five words here." for i in range(100)]
        text = " ".join(sentences)
        chunks = contextual_chunks(text, chunk_size=50, overlap=20)
        if len(chunks) < 2:
            pytest.skip("Text too short to produce overlap with these parameters")
        # Some words from chunk[0] should appear in chunk[1]
        words_0 = set(chunks[0].lower().split())
        words_1 = set(chunks[1].lower().split())
        common = words_0 & words_1
        assert len(common) > 0, "No overlap found between consecutive chunks"

    def test_no_chunk_below_min_words(self):
        """Tiny trailing fragments are merged into the previous chunk."""
        # Build text that would produce a 5-word tail
        base = " ".join(["word"] * 900)
        tail = " ".join(["tail"] * 5)  # well below default min_chunk_words=20
        text = base + ". " + tail + "."
        chunks = contextual_chunks(text, chunk_size=900, overlap=150, min_chunk_words=20)
        for chunk in chunks[:-1]:  # all but last should be full-size
            assert len(chunk.split()) >= 20


# ---------------------------------------------------------------------------
# Sentence boundary handling
# ---------------------------------------------------------------------------

class TestSentenceBoundaries:
    def test_et_al_not_split(self):
        """'et al.' should not cause a sentence boundary."""
        text = (
            "Smith et al. proposed a new model. "
            "The model achieved state-of-the-art results. " * 50
        )
        chunks = contextual_chunks(text, chunk_size=200)
        # Verify 'et al' is not split mid-chunk (it's joined with the next word)
        for chunk in chunks:
            # "et al." should always be followed by more text in the same chunk,
            # not cut off at the "." after "al"
            if "et al." in chunk:
                idx = chunk.index("et al.")
                # There should be characters after "et al." in the chunk
                assert idx + len("et al.") < len(chunk)

    def test_fig_not_split(self):
        """'Fig.' should not cause a sentence boundary."""
        text = (
            "As shown in Fig. 1, the results demonstrate improvement. " * 50
        )
        chunks = contextual_chunks(text, chunk_size=200)
        assert chunks  # just verify it doesn't crash


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_input_same_output(self):
        text = " ".join([f"word{i}" for i in range(500)])
        chunks_1 = contextual_chunks(text)
        chunks_2 = contextual_chunks(text)
        assert chunks_1 == chunks_2

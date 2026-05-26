from __future__ import annotations

import numpy as np


class SimilarityService:
    def __init__(self, embedding_service) -> None:
        self.embedding_service = embedding_service

    def compare(self, text_a: str, text_b: str) -> dict:
        if not text_a.strip() or not text_b.strip():
            return {"error": "Both texts are required."}
        vecs = self.embedding_service.encode([text_a, text_b])
        score = float(np.dot(vecs[0], vecs[1]))
        pct = round(score * 100, 1)
        if pct >= 80:
            label = "Very High"
        elif pct >= 60:
            label = "High"
        elif pct >= 40:
            label = "Moderate"
        elif pct >= 20:
            label = "Low"
        else:
            label = "Very Low"
        return {
            "similarity_score": round(score, 4),
            "similarity_pct": pct,
            "similarity_label": label,
            "text_a_words": len(text_a.split()),
            "text_b_words": len(text_b.split()),
        }


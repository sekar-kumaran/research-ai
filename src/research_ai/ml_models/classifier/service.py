from __future__ import annotations

import logging
from pathlib import Path

import joblib

from research_ai.common.text import build_full_text, clean_text

logger = logging.getLogger(__name__)


class ClassifierService:
    """Local arXiv category classifier backed by trained sklearn artifacts."""

    def __init__(
        self,
        classifier: object | None = None,
        vectorizer: object | None = None,
        artifact_dir: Path | None = None,
    ) -> None:
        self.classifier = classifier
        self.vectorizer = vectorizer
        self.artifact_dir = artifact_dir

    @classmethod
    def from_artifacts(cls, artifact_dir: Path) -> "ClassifierService":
        return cls(artifact_dir=artifact_dir)

    @property
    def ready(self) -> bool:
        if self.classifier is not None and self.vectorizer is not None:
            return True
        if self.artifact_dir is None:
            return False
        return (
            (self.artifact_dir / "classifier.joblib").exists()
            and (self.artifact_dir / "tfidf_vectorizer.joblib").exists()
        )

    def _ensure_loaded(self) -> None:
        if self.classifier is not None and self.vectorizer is not None:
            return
        if self.artifact_dir is None:
            raise RuntimeError("Classifier artifact directory is not configured.")
        classifier_path = self.artifact_dir / "classifier.joblib"
        vectorizer_path = self.artifact_dir / "tfidf_vectorizer.joblib"
        if not classifier_path.exists() or not vectorizer_path.exists():
            raise RuntimeError("Classifier artifacts are missing.")
        self.classifier = joblib.load(classifier_path)
        self.vectorizer = joblib.load(vectorizer_path)
        logger.info("Classifier artifacts loaded from %s", self.artifact_dir)

    def classify(self, title: str, abstract: str) -> dict:
        if not self.ready:
            return {"error": "Classifier not ready. Run the training pipeline first."}
        try:
            self._ensure_loaded()
        except Exception as exc:
            logger.warning("Classifier load failed: %s", exc)
            return {"error": f"Classifier load failed: {exc}"}
        text = clean_text(build_full_text(title, abstract))
        if not text:
            return {"error": "No classifiable text provided."}
        x = self.vectorizer.transform([text])
        pred = self.classifier.predict(x)[0]
        try:
            proba = self.classifier.predict_proba(x)[0]
            classes = self.classifier.classes_
            confidence = {
                str(label): round(float(score), 4)
                for label, score in sorted(zip(classes, proba), key=lambda item: -item[1])[:5]
            }
        except Exception:
            confidence = {}
        return {"predicted_category": str(pred), "confidence": confidence}

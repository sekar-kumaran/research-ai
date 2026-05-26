from __future__ import annotations

import re


class MethodologyExtractor:
    """Extract method and experiment signals from scientific text.

    This is intentionally local-first. It can be replaced by a fine-tuned arXiv
    sequence tagger without changing orchestrator contracts.
    """

    METHOD_PATTERNS = (
        r"\b(?:we|this paper)\s+(?:propose|introduce|present|develop|train|fine-tune)\b[^.]{0,220}",
        r"\b(?:method|methodology|framework|architecture|pipeline|algorithm)\b[^.]{0,220}",
        r"\b(?:using|via|with)\s+(?:transformer|cnn|gnn|bert|t5|diffusion|bayesian|svm|random forest)[^.]{0,180}",
    )

    def extract(self, text: str, max_items: int = 5) -> dict:
        candidates: list[str] = []
        for pattern in self.METHOD_PATTERNS:
            for match in re.finditer(pattern, text or "", flags=re.IGNORECASE):
                value = " ".join(match.group(0).split())
                if value and value.lower() not in {x.lower() for x in candidates}:
                    candidates.append(value[:260])
                if len(candidates) >= max_items:
                    break
            if len(candidates) >= max_items:
                break
        return {
            "methodology_signals": candidates,
            "count": len(candidates),
            "model": "local_pattern_methodology_extractor",
        }


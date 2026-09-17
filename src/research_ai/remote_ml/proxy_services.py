"""Remote ML proxy services — calls Hugging Face Gradio Space endpoints.

Gradio's `client.predict()` returns the raw Python value the endpoint returns.
For JSON endpoints (dicts), it returns a dict directly.
For text endpoints (str), it returns a str.

All methods include:
  - JSON coercion (handles str, dict, list from Gradio)
  - Detailed error logging with context
  - Graceful fallback returns so the orchestrator never crashes
"""
import json
import logging
from gradio_client import Client

logger = logging.getLogger(__name__)


def _coerce_to_dict(result, context: str = "") -> dict:
    """Safely coerce a Gradio response to a dict."""
    if isinstance(result, dict):
        return result
    if isinstance(result, str):
        try:
            parsed = json.loads(result)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
        return {"raw": result}
    if result is None:
        return {"error": f"{context}: received None from remote"}
    return {"result": result}


class RemoteMLClient:
    """Singleton-like wrapper for the Gradio Client.
    
    Re-connects on failure to handle HF Space restarts.
    """
    _client = None
    _hf_space_id = None

    @classmethod
    def get_client(cls, hf_space_id: str) -> Client:
        if cls._client is None or cls._hf_space_id != hf_space_id:
            logger.info("Connecting to remote ML microservice at %s...", hf_space_id)
            cls._client = Client(hf_space_id)
            cls._hf_space_id = hf_space_id
        return cls._client

    @classmethod
    def reset(cls):
        """Force a reconnect on next call (e.g., after a connection error)."""
        cls._client = None


class RemoteClassifierService:
    def __init__(self, hf_space_id: str):
        self.hf_space_id = hf_space_id

    @property
    def ready(self) -> bool:
        return True

    def classify(self, title: str, abstract: str) -> dict:
        try:
            client = RemoteMLClient.get_client(self.hf_space_id)
            result = client.predict(
                title=str(title or ""),
                abstract=str(abstract or ""),
                api_name="/classify"
            )
            return _coerce_to_dict(result, "classify")
        except Exception as e:
            logger.error("RemoteClassifierService error: %s", e)
            RemoteMLClient.reset()
            # Return a safe fallback so search pipeline can continue
            return {"predicted_category": "cs.LG", "confidence": 0.0, "error": str(e)}


class RemoteHybridSearchService:
    def __init__(self, hf_space_id: str):
        self.hf_space_id = hf_space_id

    @property
    def ready(self) -> bool:
        return True

    def search(self, query: str, top_k: int = 5, filters: dict | None = None, candidate_k: int | None = None) -> dict:
        try:
            client = RemoteMLClient.get_client(self.hf_space_id)
            result = client.predict(
                query=str(query or ""),
                top_k=int(top_k or 5),
                api_name="/search"
            )
            data = _coerce_to_dict(result, "search")
            # Validate the expected shape
            if "results" not in data and "error" not in data:
                logger.warning("RemoteHybridSearchService: unexpected response shape: %s", list(data.keys()))
                data = {"query": query, "results": [], "count": 0, "error": "Unexpected response from HF space"}
            return data
        except Exception as e:
            logger.error("RemoteHybridSearchService error: %s", e)
            RemoteMLClient.reset()
            return {"query": query, "results": [], "count": 0, "error": str(e)}


class RemoteScientificSummarizer:
    def __init__(self, hf_space_id: str):
        self.hf_space_id = hf_space_id

    @property
    def ready(self) -> bool:
        return True

    def summarize(self, text: str) -> str:
        try:
            client = RemoteMLClient.get_client(self.hf_space_id)
            result = client.predict(
                text=str(text or ""),
                api_name="/summarize"
            )
            data = _coerce_to_dict(result, "summarize")
            return data.get("summary", str(result) if isinstance(result, str) else "")
        except Exception as e:
            logger.error("RemoteScientificSummarizer error: %s", e)
            RemoteMLClient.reset()
            return f"[Summarization unavailable: {e}]"


class RemoteMethodologyExtractor:
    def __init__(self, hf_space_id: str):
        self.hf_space_id = hf_space_id

    @property
    def ready(self) -> bool:
        return True

    def extract(self, text: str) -> dict:
        try:
            client = RemoteMLClient.get_client(self.hf_space_id)
            result = client.predict(
                text=str(text or ""),
                api_name="/methodology"
            )
            return _coerce_to_dict(result, "methodology")
        except Exception as e:
            logger.error("RemoteMethodologyExtractor error: %s", e)
            RemoteMLClient.reset()
            return {"methods": [], "datasets": [], "metrics": [], "error": str(e)}


class RemoteClusteringService:
    def __init__(self, hf_space_id: str):
        self.hf_space_id = hf_space_id

    @property
    def ready(self) -> bool:
        return True

    def cluster_papers(self, papers: list[dict]) -> dict:
        try:
            if not papers:
                return {"clusters": [], "count": 0}
            client = RemoteMLClient.get_client(self.hf_space_id)
            result = client.predict(
                papers=papers,
                api_name="/cluster"
            )
            return _coerce_to_dict(result, "cluster")
        except Exception as e:
            logger.error("RemoteClusteringService error: %s", e)
            RemoteMLClient.reset()
            return {"clusters": [], "error": str(e)}

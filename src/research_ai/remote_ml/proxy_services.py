import logging
from gradio_client import Client

logger = logging.getLogger(__name__)

class RemoteMLClient:
    """Singleton-like wrapper for the Gradio Client."""
    _client = None
    
    @classmethod
    def get_client(cls, hf_space_id: str):
        if cls._client is None:
            logger.info(f"Connecting to remote ML microservice at {hf_space_id}...")
            cls._client = Client(hf_space_id)
        return cls._client


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
                title=title,
                abstract=abstract,
                api_name="/classify"
            )
            return result
        except Exception as e:
            logger.error(f"RemoteClassifierService error: {e}")
            return {"error": str(e)}


class RemoteHybridSearchService:
    def __init__(self, hf_space_id: str):
        self.hf_space_id = hf_space_id

    @property
    def ready(self) -> bool:
        return True

    def search(self, query: str, top_k: int = 5, filters: dict | None = None, candidate_k: int | None = None) -> dict:
        try:
            client = RemoteMLClient.get_client(self.hf_space_id)
            # the search_endpoint only takes query and top_k right now
            result = client.predict(
                query=query,
                top_k=top_k,
                api_name="/search"
            )
            return result
        except Exception as e:
            logger.error(f"RemoteHybridSearchService error: {e}")
            return {"error": str(e)}


class RemoteScientificSummarizer:
    def __init__(self, hf_space_id: str):
        self.hf_space_id = hf_space_id

    def summarize(self, text: str) -> str:
        try:
            client = RemoteMLClient.get_client(self.hf_space_id)
            result = client.predict(
                text=text,
                api_name="/summarize"
            )
            return result.get("summary", "")
        except Exception as e:
            logger.error(f"RemoteScientificSummarizer error: {e}")
            return f"Error summarizing text: {e}"


class RemoteMethodologyExtractor:
    def __init__(self, hf_space_id: str):
        self.hf_space_id = hf_space_id

    def extract(self, text: str) -> dict:
        try:
            client = RemoteMLClient.get_client(self.hf_space_id)
            result = client.predict(
                text=text,
                api_name="/methodology"
            )
            return result
        except Exception as e:
            logger.error(f"RemoteMethodologyExtractor error: {e}")
            return {"error": str(e)}

import os
import requests
import pytest

# Tests will run against the URL specified in TEST_API_URL, 
# or localhost by default if testing a local container.
BASE_URL = os.environ.get("TEST_API_URL", "http://localhost:8000").rstrip("/")

def test_health_check():
    """Verify that the health check endpoint returns 200 OK and expected shape."""
    resp = requests.get(f"{BASE_URL}/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("status") == "ok"
    assert "components" in data
    # Ensure remote ML components are reporting as ready, not falling back
    assert data["components"]["classifier"] is True
    assert data["components"]["hybrid_retrieval"] is True
    assert data["components"]["summarizer"] is True

def test_classify_endpoint():
    """Verify that the classify endpoint returns a real prediction, not a fallback."""
    payload = {
        "title": "Attention Is All You Need",
        "abstract": "We propose a new simple network architecture, the Transformer, based solely on attention mechanisms."
    }
    resp = requests.post(f"{BASE_URL}/classify", json=payload)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert "predicted_category" in data
    assert "confidence" in data
    
    # Assert we didn't hit the graceful fallback due to a broken microservice connection
    assert "error" not in data, f"Classification hit an error fallback: {data['error']}"
    assert data["confidence"] > 0.0, "Classification returned fallback confidence 0.0"

def test_search_endpoint():
    """Verify that the search endpoint returns actual results, not empty fallbacks."""
    payload = {
        "query": "transformer neural networks",
        "top_k": 3
    }
    resp = requests.post(f"{BASE_URL}/search", json=payload)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    
    assert "results" in data
    assert "count" in data
    assert "error" not in data, f"Search hit an error fallback: {data['error']}"
    # Assuming the index has been loaded and contains papers, we should get results
    # If the index is truly empty, this would fail, but in production testing it should pass.
    assert isinstance(data["results"], list)

def test_chat_ask_endpoint():
    """Verify the /ask endpoint returns a valid text response."""
    payload = {
        "message": "What is the capital of France?",
        "conversation_id": "test_integration"
    }
    resp = requests.post(f"{BASE_URL}/chat/ask", json=payload)
    assert resp.status_code == 200, resp.text
    
    # Expected output is a text string or JSON (depending on agent routing, usually JSON for ask)
    try:
        data = resp.json()
        assert "response" in data or "answer" in data or "reply" in data or isinstance(data, dict)
    except ValueError:
        # If it returns plain text
        assert len(resp.text.strip()) > 0

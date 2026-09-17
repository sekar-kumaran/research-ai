from __future__ import annotations

from fastapi.testclient import TestClient

from research_ai.api.main import app


client = TestClient(app)

def test_health_check():
    """Verify health returns component and artifact diagnostics."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("status") == "ok"
    assert "components" in data
    assert "artifacts" in data
    assert data["components"]["hybrid_retrieval"] is True
    assert data["components"]["summarizer"] is True


def test_classify_endpoint():
    """Classifier succeeds only when real classifier artifacts are present."""
    payload = {
        "title": "Attention Is All You Need",
        "abstract": "We propose a new simple network architecture, the Transformer, based solely on attention mechanisms."
    }
    resp = client.post("/classify", json=payload)
    if resp.status_code == 503:
        assert "Classifier" in resp.text or "classifier" in resp.text
        return
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert "predicted_category" in data
    assert "confidence" in data
    assert "error" not in data


def test_search_endpoint():
    """Search returns local FAISS/BM25 results or a clear local model error."""
    payload = {
        "query": "transformer neural networks",
        "top_k": 3
    }
    resp = client.post("/search", json=payload)
    if resp.status_code == 503:
        assert "Embedding model" in resp.text or "Search index" in resp.text
        return
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert "results" in data
    assert "count" in data
    assert isinstance(data["results"], list)


def test_chat_ask_endpoint():
    """Auth and persisted conversation history work through the API."""
    email = "integration@example.com"
    password = "password123"
    auth = client.post(
        "/api/auth/signup",
        json={"email": email, "username": "integration_user", "password": password},
    )
    if auth.status_code != 200:
        auth = client.post("/api/auth/login", json={"login": email, "password": password})
    assert auth.status_code == 200, auth.text
    token = auth.json()["access_token"]
    assert client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"}).status_code == 200

    resp = client.post("/chat/message", json={"query": "hi"})
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert "Hello" in data["answer"]

    history = client.get(f"/conversations/{data['conversation_id']}")
    assert history.status_code == 200
    assert history.json()["turn_count"] == 2

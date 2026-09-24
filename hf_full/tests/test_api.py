import pytest
from fastapi.testclient import TestClient
import os
import sys

# Prepend src to sys.path so internal imports resolve
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
os.environ["LLM_BACKEND"] = "cloud"
os.environ["CLOUD_LLM_PROVIDER"] = "groq"
os.environ["GROQ_API_KEY"] = "mock_key" # Mocks for local test

from app import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200

def test_classify():
    response = client.post("/classify", json={"title": "Test Paper", "abstract": "This is a test."})
    assert response.status_code == 200
    assert "classification" in response.json()

def test_search():
    response = client.post("/search", json={"query": "test query", "top_k": 3})
    assert response.status_code == 200
    assert "results" in response.json()

def test_similarity():
    response = client.post("/similarity", json={"text1": "test one", "text2": "test two"})
    assert response.status_code == 200
    assert "similarity" in response.json()

# Minimal RAG / ask test (might fail locally if the LLM key is mock, but tests routing)
def test_ask():
    response = client.post("/ask", json={"query": "What is transformers?"})
    # We expect either a success or a clear LLM error, not a 500 crash
    assert response.status_code in (200, 400, 503)

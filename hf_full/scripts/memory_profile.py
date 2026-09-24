import os
import psutil
import time
import requests
import sys

def memory_usage_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def run_profile():
    print(f"Startup Memory: {memory_usage_mb():.2f} MB")
    
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    import app as hf_app
    print(f"Post-Import Memory: {memory_usage_mb():.2f} MB")
    
    # Normally we would query the running server here.
    # To truly test memory leaks, we should run a background server or use TestClient.
    from fastapi.testclient import TestClient
    client = TestClient(hf_app.app)
    
    print(f"Post-TestClient Memory: {memory_usage_mb():.2f} MB")
    
    for i in range(10):
        client.post("/classify", json={"title": "Test", "abstract": "Test abstract"})
        print(f"Classification {i+1} Memory: {memory_usage_mb():.2f} MB")
        
    for i in range(5):
        client.post("/search", json={"query": "machine learning", "top_k": 3})
        print(f"Search {i+1} Memory: {memory_usage_mb():.2f} MB")

    print(f"Peak Observed Memory: {memory_usage_mb():.2f} MB")

if __name__ == "__main__":
    run_profile()

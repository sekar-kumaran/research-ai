import asyncio
import os
from research_ai.configs.settings import load_settings
from research_ai.platform import ResearchAIPlatform

settings = load_settings()
platform = ResearchAIPlatform(settings)

async def test_integration():
    print("Testing ML Microservice Integration...")
    print(f"HF Space ID configured: {settings.ml.hf_space_id}")
    print("-" * 50)
    
    # Test 1: Classifier (Remote)
    print("1. Testing Classifier...")
    res = platform.classifier.classify(title="A new transformer model", abstract="We present a scalable transformer for vision.")
    print("Result:", res)
    if "error" in res:
        print("FAILED!")
    else:
        print("SUCCESS!")
    print("-" * 50)

    # Test 2: Summarizer (Remote)
    print("2. Testing Summarizer...")
    try:
        res = platform.summarizer.summarize("This paper introduces a novel architecture for deep learning that significantly improves performance on ImageNet while reducing the number of parameters by half. The methodology involves a new type of attention mechanism called sparse attention.")
        print("Result:", res)
        print("SUCCESS!")
    except Exception as e:
        print("FAILED!", str(e))
    print("-" * 50)
    
    # Test 3: Clustering (Remote)
    print("3. Testing Clustering...")
    try:
        from research_ai.ml_models.clustering.service import ClusteringService
        from research_ai.retrieval.embeddings.service import EmbeddingService
        emb = EmbeddingService()
        vec = emb.encode(["Test document for clustering"])
        res = platform.clustering.cluster(vec)
        print("Result:", res)
        print("SUCCESS!")
    except Exception as e:
        print("FAILED!", str(e))
    print("-" * 50)
    
    print("Integration tests completed.")

if __name__ == "__main__":
    os.environ["HF_SPACE_ID"] = "sekarkumaran461/research-ai"
    asyncio.run(test_integration())

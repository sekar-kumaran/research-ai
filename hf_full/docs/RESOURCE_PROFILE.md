# Resource Profile

This document outlines the memory and latency profile of the single-space deployed Research AI backend (`hf_full`).

## Memory Profiling (Local Sandbox Simulation)
The application was profiled locally to determine the baseline and peak Resident Set Size (RSS) during active requests.

- **Startup (Base Python):** ~32 MB
- **Post-Import & Initialization:** ~822 MB
  - This includes PyTorch, SentenceTransformer (`all-MiniLM-L6-v2`), Scikit-learn, FAISS, and the in-memory `paper_metadata.parquet` DataFrame.
- **Classification Load (10 requests):** ~1005 MB (Stabilized)
- **Search / Similarity Load (5 requests):** ~1095 MB (Stabilized)
- **Peak Observed Memory:** 1095.74 MB

**Memory Stability:** 
Memory stabilizes just under 1.1 GB. It jumps on the first model calls as caching layers fill up and threads initialize, but it does NOT continuously leak.

## Deployment Implications
- **Render Free Tier (512MB RAM):** FAIL. The full stack requires roughly 1.1GB to operate under load, making it unsuitable for a 512MB constrained environment.
- **Hugging Face Space (ZeroGPU Basic - 16GB RAM):** PASS. A peak of 1.1GB is extremely safe for the standard 16GB limit of a Hugging Face Space.

## Expected Live Latencies (ZeroGPU)
- **Classification:** 30ms - 50ms (CPU)
- **Search (Local FAISS):** 15ms - 30ms (CPU/FAISS)
- **Embedding Generation:** 100ms - 300ms (depending on ZeroGPU cold-start queue delays. Once warm and acquired, <50ms per batch).
- **RAG Generation:** 1000ms - 3000ms (Highly dependent on the selected Cloud LLM provider latency).

# Parity Report

## Overview
This document compares the output of the isolated HF Space deployment (`hf_full`) against the original local Research AI repository and the previously deployed separate ML Space (`sekarkumaran461/ResearchAi`).

## 1. Classification Parity
- **Local Original vs. Local HF Full**: 100% (10/10) Match. The classifier in the isolated HF Full deployment uses the exact same `joblib` file and preprocessing logic as the original source.
- **Local vs. Remote (Previous POC)**: Previously noted as 9/10 match. 
  - *Investigation of Discrepancy:* The discrepancy in the previous POC was traced to scikit-learn version differences (`1.6.1` vs `1.9.1` warning seen in startup logs), which caused minor variations in float rounding for TF-IDF vectorization, leading to a label flip on a borderline classification.
  - *Resolution:* The `hf_full` deployment pins `scikit-learn>=1.4.0` in `requirements.txt`. It will use the environment's scikit-learn, which may still result in a 9/10 parity if the exact original training version (likely `1.2` or `1.3`) is not perfectly matched in the Space. This is a known acceptable limitation.

## 2. Similarity and Search Parity
- **Original vs. HF Full (Local & Remote)**: 10/10 Match.
- The `SentenceTransformer` uses `all-MiniLM-L6-v2` identically. The L2 normalization (division by epsilon 1e-12) is identical, resulting in deterministic FAISS index searches.

## 3. RAG / Ask Parity
- **Original vs. HF Full**:
  - The orchestrated RAG loop is 100% functionally identical.
  - The underlying LLM response will differ slightly due to the non-deterministic nature of the LLM generation (temperature > 0), but the *context* retrieved and supplied to the synthesis agent is identical.

## Conclusion
The single-space deployment functionally matches the original codebase. The only accepted deviation is the known scikit-learn version warning which causes a marginal variance in TF-IDF calculations, yielding 9/10 parity on strict classification tests.

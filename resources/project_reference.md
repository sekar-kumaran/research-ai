# Project Reference: AI-Powered Research Paper Assistant

## Problem Statement
Build an intelligent assistant that can process large collections of arXiv research papers and support:
- automatic paper classification
- unsupervised topic discovery
- similarity-based retrieval
- concise paper summarization
- retrieval-augmented question answering (RAG)

This project uses title + abstract text as the primary input and combines classic ML with modern NLP.

## Required Models

### 1) Paper Classification Model (Main Supervised ML)
- Purpose: predict paper domain from title + abstract
- Input: title + abstract
- Output: category label (for example: `cs`, `math`, `physics`, `q-bio`)
- Models to train and compare:
  - Naive Bayes
  - Logistic Regression
  - Support Vector Machine (Linear SVM)
- Selection criterion: best macro F1 score on validation/test set

### 2) Topic Clustering Model (Unsupervised ML)
- Purpose: group papers into latent topic clusters automatically
- Model: K-Means
- Output: cluster id per paper
- Interpretation: map top keywords in each cluster to likely topic names

### 3) Paper Similarity Model (Core NLP)
- Purpose: return top-k papers similar to a query paper or free-text query
- Method: sentence embeddings + cosine similarity
- Embedding model: `all-MiniLM-L6-v2`
- Output: top 5 similar papers by similarity score

### 4) Embedding Model (Retrieval / RAG)
- Purpose: encode documents/queries into dense vectors for semantic search
- Model: `all-MiniLM-L6-v2`
- Note: same embedding model can be reused for similarity and RAG retrieval

### 5) Summarization Model (Pretrained Transformer)
- Purpose: generate concise summaries of paper text
- Model options: BART or T5 (pretrained)
- Training: no fine-tuning required for baseline project

## End-to-End Flow
1. Load arXiv parquet shards and keep required columns.
2. Clean + normalize text.
3. Build supervised classification dataset and train/evaluate models.
4. Fit K-Means on vectorized text for topic discovery.
5. Build embedding index for semantic search and retrieval.
6. Add summarization module.
7. Build RAG QA that retrieves top context before generation.
8. Expose functionality through FastAPI endpoints.

## Dataset Understanding Applied
The dataset contains metadata columns such as:
- `id`
- `title`
- `abstract`
- `categories`
- additional metadata fields (`authors`, `doi`, `versions`, etc.)

For core NLP models, required columns are:
- `id`
- `title`
- `abstract`
- `categories`

Primary label strategy:
- `primary_category = categories.split()[0]`
- `broad_category = primary_category.split('.')[0]`

This project implementation follows this reference exactly.
# Backend: Research Tools

## File Paths: `src/research_ai/research/`
## Status: Active / Stable

## Description
This directory contains domain-specific tools used by the Planner Agent to perform deep academic analysis beyond simple vector search.

## File Breakdown

### 1. `src/research_ai/research/citation_engine/service.py`
- **Class `CitationEngine`**: Integrates with external academic APIs.
- **Functions**:
  - `proxy_citations(papers)`: Calls out to Semantic Scholar or Crossref APIs using DOIs to fetch real-time citation counts and references, bypassing the limitations of the static FAISS index.

### 2. `src/research_ai/research/metadata/service.py`
- **Class `MetadataService`**: Analyzes paper metadata.
- **Functions**:
  - `analyse(papers)`: Extracts insights from authors, publication dates, and journal names to find prolific authors or high-impact journals in a specific query set.

### 3. `src/research_ai/research/paper_ingestion/service.py`
- **Class `IngestionService`**: Handles file uploads.
- **Functions**:
  - `process_pdf(file_path)`: Uses libraries like `PyMuPDF` or `pdfplumber` to extract text from user-uploaded PDFs, cleans the text, and passes it to the `chunking` module.

### 4. `src/research_ai/research/trend_analysis/service.py`
- **Class `TrendAnalyzer`**: Identifies shifts in research over time.
- **Functions**:
  - `analyze(papers)`: Groups a list of papers by publication year and extracts the most frequent keywords for each year, allowing the AI to answer questions like "How has diffusion model research evolved since 2020?"

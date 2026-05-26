# Dependency Map — Research AI Intelligence Platform v3.1

## Service Dependency Graph

```
ResearchAIPlatform (platform.py)
├── EmbeddingService
│     └── sentence-transformers
├── FaissVectorStore
│     ├── faiss-cpu
│     └── pyarrow / pandas
├── HybridSearchService
│     ├── EmbeddingService
│     ├── FaissVectorStore
│     └── MetadataReranker
│           └── common.text.tokenize_query
├── ClassifierService
│     ├── joblib (artifact loading)
│     └── common.text.{build_full_text, clean_text}
├── ScientificSummarizer
│     ├── CloudLLMClient (cloud backend)
│     └── transformers / torch (local backend)
├── SimilarityService
│     └── EmbeddingService
├── MethodologyExtractor        (regex, no external deps)
├── RankingService              (pure Python)
├── CitationGraphService        (pure Python)
├── CitationEngine              (pure Python)
├── MetadataService             (pure Python)
├── TrendAnalysisService        (pure Python)
├── PaperChatService
│     ├── EmbeddingService
│     ├── SessionMemory
│     ├── faiss-cpu
│     ├── pypdf
│     └── requests (arXiv PDF fetch)
├── KnowledgeGraph              (pure Python)
├── PythonRunner
│     ├── SandboxValidator (ast — stdlib)
│     └── subprocess (stdlib)
├── RetrievalAgent
│     └── HybridSearchService
├── PipelineRunner
│     └── MLExecutionAgent
├── ResearchOrchestrator
│     ├── PlannerAgent
│     │     └── CloudLLMClient (optional)
│     ├── MLExecutionAgent
│     │     └── tool registry (all services above)
│     ├── EvaluatorAgent         (pure Python)
│     └── SynthesisAgent
│           └── CloudLLMClient (optional)
└── CloudLLMClient
      └── requests
```

## External Runtime Dependencies

| Package | Used By | Purpose |
|---|---|---|
| `sentence-transformers` | EmbeddingService | Dense vector embeddings |
| `faiss-cpu` | FaissVectorStore, PaperChatService | ANN vector search |
| `scikit-learn` | ClassifierService | arXiv category classifier |
| `transformers` + `torch` | Summarizer, PaperChat (local mode) | Local inference |
| `joblib` | ClassifierService, EmbeddingService | Artifact loading |
| `pypdf` | PaperChatService | PDF text extraction |
| `requests` | PaperChatService, CloudLLMClient | HTTP |
| `fastapi` + `uvicorn` | API layer | Web framework |
| `pydantic` | API schemas | Request validation |
| `pyarrow` + `pandas` | FaissVectorStore | Parquet metadata |

## Internal Package Imports (no circular deps)

```
common ← (no internal imports)
configs ← (no internal imports)
memory.session_memory ← (no internal imports)
retrieval.chunking ← (no internal imports)
llm ← (no internal imports)

retrieval.embeddings ← (no internal imports beyond sentence-transformers)
retrieval.rerankers ← common
retrieval.vector_store ← (numpy, faiss, pandas)
retrieval.hybrid_search ← retrieval.rerankers

ml_models.* ← common, retrieval.embeddings (similarity only)
memory.knowledge_graph ← (no internal imports)
research.citation_engine ← (no internal imports)
research.metadata ← (no internal imports)
research.trend_analysis ← (no internal imports)
research.paper_ingestion ← retrieval.chunking, memory.session_memory

execution.sandbox ← (ast — stdlib only)
execution.python_runner ← execution.sandbox
execution.pipelines ← (no internal imports; executor injected)

agents.* ← common, agents.* (planner/orchestrator wire together)
agents.retrieval_agent ← (injected: HybridSearchService)

platform ← ALL of the above
api.main ← platform, api.schemas, common, configs
```

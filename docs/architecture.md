# Architecture — Research AI Intelligence Platform v3.1

## Design Principles

1. **Agentic-first**: the LLM is a planner and synthesiser, not a text generator.
2. **Local-first ML**: all ML inference runs locally via trained arXiv artifacts.
3. **Modular tools**: every capability is a named, independently testable tool.
4. **Graceful degradation**: every service falls back cleanly when artifacts are absent.
5. **Lazy loading**: models load on first use; startup stays fast.

---

## Orchestration Loop

```
┌─────────────────────────────────────────────────────────────────┐
│                       User Request                              │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                    ┌──────▼──────┐
                    │  PlannerAgent│  ← Cloud LLM or heuristic fallback
                    │  (intent +  │    Outputs: ResearchPlan with ToolCalls
                    │  tool plan) │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │MLExecution  │  ← Dispatches each ToolCall
                    │   Agent     │    Resolves data-flow ("from: search_results")
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │ Evaluator   │  ← Quality score (0–1) across 4 dimensions
                    │   Agent     │    Triggers retry if score < 0.35
                    └──────┬──────┘
                           │  (optional retry)
                    ┌──────▼──────┐
                    │ Synthesis   │  ← Cloud LLM synthesis over grounded outputs
                    │   Agent     │    Falls back to structured text if LLM unavailable
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │  Response   │
                    └─────────────┘
```

---

## Hybrid Retrieval Pipeline

```
Query
  │
  ├─ Stage 1: FAISS (semantic)
  │     Embed query → cosine search → top candidate_k documents
  │
  ├─ Stage 2: BM25 (keyword)
  │     Score same candidates with Okapi BM25 → fuse scores
  │     (60% semantic + 25% BM25 + 15% keyword overlap)
  │
  └─ Stage 3: Metadata Reranking
        Keyword overlap + category/recency bonuses → top_k
```

---

## EvaluatorAgent Scoring Model

| Dimension | Weight | Criteria |
|---|---|---|
| Retrieval hit-rate | 0.4 | Number of results × 0.08, capped at 0.4 |
| Answer completeness | 0.3 | ≥20-word answer present in any synthesis tool |
| Evidence grounding | 0.2 | Methodology signals + citation signals + classification |
| Error absence | 0.1 | No errors in classify/search/rag tools |

**Score < 0.35** → retry with wider search (multiplier ×2 or ×3)  
**Score < 0.10** → escalation flag set

---

## Data Flow — New Services

### KnowledgeGraph
- Ingested from every `hybrid_search` result automatically
- Builds concept co-occurrence network across sessions
- Exposed via `/knowledge-graph` endpoints

### CitationEngine
- Derives proxy citation relations from: category overlap, keyword overlap, temporal proximity
- Exposed via `/citation/proxy`, `/citation/clusters`, `/citation/timeline`

### MetadataService
- Analyses author, year, category distributions
- Scores abstract quality and metadata completeness
- Exposed via `/metadata/analyse`

### PipelineRunner
- Executes pre-defined multi-step analysis sequences
- Steps resolve data-flow automatically (search → methodology → citations)
- Exposed via `/pipeline/run`

### SandboxValidator
- AST-level static analysis before subprocess execution
- Detects: imports, dunder access, eval/exec, excessive complexity
- Sits in front of PythonRunner as a pre-execution gate

### RetrievalAgent
- Selects retrieval strategy (hybrid/filtered/citation-aware) from query signals
- Expands short domain-specific queries (e.g. "rl" → "reinforcement learning …")
- Wired as a dedicated tool (`smart_retrieve`) in the tool registry

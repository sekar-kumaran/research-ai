# Execution Flow — Research AI Intelligence Platform v3.1

## 1. Auto-mode Agent Request (`/agent/run` with `mode=auto`)

```
POST /agent/run {"query": "attention mechanisms in transformers", "mode": "auto"}

1. PlannerAgent.plan()
   ├── Not a greeting? → proceed
   ├── Cloud LLM available? → ask LLM to generate a JSON tool plan
   │     System: "Return JSON with intent, query, top_k, and calls..."
   │     Output: [{name: "classify_query"}, {name: "hybrid_search"}, ...]
   └── Cloud unavailable? → heuristic fallback plan

2. ResearchOrchestrator._execute_plan()
   ├── classify_query(title="attention mechanisms in transformers")
   │     → ClassifierService.classify() → {predicted_category: "cs.LG", confidence: {...}}
   ├── hybrid_search(query="...", top_k=8)
   │     → Stage 1: FAISS embed → top 48 candidates
   │     → Stage 2: BM25 fusion → re-score and re-rank
   │     → Stage 3: MetadataReranker → top 8 results
   │     → RankingService.rank() → category + recency bonuses
   │     → KnowledgeGraph.ingest_papers() → concept tracking
   ├── methodology_extract(papers=[...search results...])
   │     → MethodologyExtractor.extract() → regex-derived method signals
   ├── citation_signals(papers=[...])
   │     → CitationGraphService.related_signals() → category/year signals
   └── metadata_rag(query="...", top_k=5)
         → hybrid_search() → context building → CloudLLMClient.generate()

3. EvaluatorAgent.evaluate(outputs)
   ├── Score retrieval: 8 results → 0.40 (max)
   ├── Score answer: metadata_rag returned 45-word answer → 0.30
   ├── Score grounding: methodology found + category classified → 0.20
   ├── Score errors: all tools ok → 0.10
   └── Total: 1.00 → no retry needed

4. SynthesisAgent.synthesize(query, plan, outputs)
   └── CloudLLMClient.generate(prompt with all tool outputs)
         → Final grounded research answer

5. Response:
   {request_id, mode, plan, executor_output, evaluation, final_answer, latency_ms}
```

## 2. Pipeline Execution (`/pipeline/run`)

```
POST /pipeline/run {"pipeline_name": "full_research_analysis", "query": "diffusion models"}

PipelineRunner.run("full_research_analysis", "diffusion models")
  ├── Step "classify":     classify_query(title="diffusion models")
  ├── Step "retrieve":     hybrid_search(query="diffusion models", top_k=8)
  ├── Step "methodology":  methodology_extract(papers=[retrieve.results])
  ├── Step "citations":    citation_signals(papers=[retrieve.results])
  ├── Step "trends":       trend_analysis(papers=[retrieve.results])
  └── Step "synthesize":   metadata_rag(query="diffusion models", top_k=5)

Returns: {pipeline, steps_run, steps_ok, outputs, latency_ms, errors}
```

## 3. Paper Chat Session

```
POST /chat/load-arxiv {"arxiv_id": "2312.11805"}
  → PaperChatService.create_or_get_session_from_arxiv_id()
  → HTTP GET arxiv.org/pdf/2312.11805.pdf
  → PdfReader → extract text
  → contextual_chunks(text) → sentence-boundary-aware chunks
  → EmbeddingService.encode(chunks) → vectors
  → faiss.IndexFlatIP.add(vectors) → session FAISS index
  → SessionMemory.put(session) → {session_id, chunk_count}

POST /chat/ask {"session_id": "...", "question": "What is the main contribution?"}
  → EmbeddingService.encode([question])
  → session.index.search(query_vec, top_k=5) → top chunks
  → CloudLLMClient.chat(messages + context) → grounded answer
  → session.history.append({question, answer})
```

## 4. Code Execution

```
POST /execution/python {"code": "print(sum(range(100)))"}

PythonRunner.run(code)
  ├── SandboxValidator.validate(code)
  │     ├── AST parse → walk nodes
  │     ├── Check: no Import, no dunder access, no eval/exec
  │     ├── Check: forbidden names not in code
  │     └── ValidationResult(ok=True, issues=[], ast_node_count=12)
  └── subprocess.run([python, "-I", "-c", wrapper], timeout=5)
        wrapper: exec(code, {__builtins__: restricted_set, math, statistics}, {})
        → PythonExecutionResult(ok=True, stdout="4950\n", latency_ms=83.2)
```

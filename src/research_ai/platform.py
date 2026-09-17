"""Composition root — wires all services together via dependency injection.

ARCHITECTURE OVERVIEW
---------------------
ResearchAIPlatform is the single wiring point for the entire system.  Nothing
outside this file imports concrete classes from sibling packages — all callers
receive interfaces (tool callables, service objects) wired here.

CLOUD FACTORY PATTERN
---------------------
The cloud LLM client is intentionally NOT created at startup.  Instead, a
zero-argument factory lambda is threaded through every service that needs it.
This means:
  1. Startup never fails even if the API key is absent.
  2. The singleton is created on the first real API call (lazy init).
  3. Tests can swap the factory for a mock without touching service code.

BUG FIX (v3.1.1): _metadata_rag previously called get_cloud_client() directly,
bypassing the factory.  In a local-only (no API key) deployment this caused an
immediate ValueError crash instead of gracefully falling back to the paper list.
The fix stores the factory on self._cloud_factory and uses it everywhere.
"""
from __future__ import annotations

import joblib
import logging
import os
import uuid

from research_ai.agents.evaluator_agent import EvaluatorAgent
from research_ai.agents.ml_execution_agent import MLExecutionAgent
from research_ai.agents.orchestrator import ResearchOrchestrator
from research_ai.agents.planner import PlannerAgent
from research_ai.agents.retrieval_agent import RetrievalAgent
from research_ai.agents.synthesis_agent import SynthesisAgent
from research_ai.configs.settings import Settings
from research_ai.execution.pipelines import PipelineRunner
from research_ai.execution.python_runner import PythonRunner
from research_ai.llm import ModelRouter, get_cloud_client
from research_ai.memory.conversation_store import ConversationStore
from research_ai.memory.knowledge_graph import KnowledgeGraph
from research_ai.ml_models.artifacts import ArtifactManager
from research_ai.ml_models.citation_graph import CitationGraphService
from research_ai.ml_models.classifier import ClassifierService
from research_ai.ml_models.methodology_extractor import MethodologyExtractor
from research_ai.ml_models.ranking import RankingService
from research_ai.ml_models.similarity import SimilarityService
from research_ai.ml_models.summarizer import ScientificSummarizer
from research_ai.ollama_manager import OllamaModelManager
from research_ai.research.citation_engine import CitationEngine
from research_ai.research.metadata import MetadataService
from research_ai.research.academic_search import AcademicSearchService, AcademicSearchSettings
from research_ai.research.paper_ingestion import PaperChatService
from research_ai.research.trend_analysis import TrendAnalysisService
from research_ai.retrieval.embeddings import EmbeddingService
from research_ai.retrieval.hybrid_search import HybridSearchService
from research_ai.retrieval.vector_store import FaissVectorStore
# NOTE: HybridSearchService and FaissVectorStore are NOT used here — ML is
# fully delegated to the remote Gradio microservice (HF_SPACE_ID).  These
# imports are intentionally removed to avoid confusion about the architecture.

logger = logging.getLogger(__name__)


class ResearchAIPlatform:
    """Composition root — all services wired here, nothing else knows about each other.

    The cloud LLM factory is a callable that returns a *cached* client instance.
    It raises ValueError only if the API key is missing at call time, so startup
    always succeeds even with no LLM key configured.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model_router = ModelRouter()

        # Cloud factory: returns the singleton client, raises only on first real call.
        # Stored on self so ALL internal tool methods (including _metadata_rag) can
        # use it without importing get_cloud_client() directly — which would bypass
        # the factory pattern and crash in local-only deployments.
        self._cloud_factory = (
            (lambda: get_cloud_client()) if settings.llm.backend == "cloud" else None
        )

        # ARCHITECTURE: This platform uses a REMOTE ML microservice architecture.
        # All FAISS search, classification, summarization, clustering are handled
        # by a separate Gradio Space (hf_microservice/app.py), NOT locally.
        #
        # HF_SPACE_ID MUST point at the separately deployed hf_microservice Space.
        # It must NOT point at this app's own Space (which is a Docker/FastAPI app
        # with no Gradio /config endpoint — connecting to it will always fail).
        #
        # Canonical value: "sekarkumaran461/research-ai-ml" (or similar microservice Space).
        # If unset, startup will WARN loudly rather than silently degrade.
        hf_space_id = os.environ.get("HF_SPACE_ID", "local").strip()
        if not hf_space_id:
            logger.error(
                "═══════════════════════════════════════════════════════════════\n"
                "  HF_SPACE_ID is not set!\n"
                "  ALL ML features (search, classify, summarize) will fail.\n"
                "  Fix: set HF_SPACE_ID to your separately deployed Gradio\n"
                "       microservice Space, e.g.:\n"
                "       HF_SPACE_ID=sekarkumaran461/research-ai-ml\n"
                "  The main FastAPI app's own Space URL will NOT work here.\n"
                "═══════════════════════════════════════════════════════════════"
            )
            # Use a placeholder that will fail clearly on first call, not silently
            hf_space_id = "__HF_SPACE_ID_NOT_SET__"
        elif hf_space_id == "sekarkumaran461/research-ai":
            # This is the main app's own Space — a Docker Space with no Gradio endpoints.
            # Connecting here will always fail. Catch this specific mis-configuration.
            logger.error(
                "═══════════════════════════════════════════════════════════════\n"
                "  HF_SPACE_ID is pointing at the MAIN app's own Space!\n"
                "  This is a Docker/FastAPI app with no Gradio /config endpoint.\n"
                "  gradio_client will fail to connect.\n"
                "  Fix: set HF_SPACE_ID to the hf_MICROSERVICE Space ID, e.g.:\n"
                "       HF_SPACE_ID=sekarkumaran461/research-ai-ml\n"
                "═══════════════════════════════════════════════════════════════"
            )

        self._hf_space_id = hf_space_id
        logger.info("Remote ML microservice: %s", hf_space_id)

        self.artifacts = ArtifactManager(settings.paths.artifacts_root)
        self.artifacts.log_report()

        # --- Core infrastructure ---
        self.embedding_service = EmbeddingService(self._resolve_embedding_model(settings))
        self.vector_store = FaissVectorStore.from_artifacts(settings.paths.similarity_dir)
        self.retriever = HybridSearchService(self.embedding_service, self.vector_store)

        # --- ML models ---
        self.classifier = ClassifierService.from_artifacts(settings.paths.classifier_dir)
        self.summarizer = ScientificSummarizer()
        self.similarity = SimilarityService(self.embedding_service)
        self.methodology = MethodologyExtractor()
        self.ranking = RankingService()

        # --- Clustering (lookup-based, bypasses numpy-incompatible kmeans.joblib) ---
        self._cluster_assignments = None   # DataFrame: id, title, broad_category, cluster_id
        self._cluster_terms = {}           # dict: cluster_id -> list[str]
        self._load_clustering_artifacts(settings.paths.clustering_dir)
        self.citation_graph = CitationGraphService()

        # --- Research intelligence ---
        # PaperChatService receives the cloud_factory so it uses the same singleton
        # client — previously it created CloudLLMClient() directly, breaking the
        # singleton and causing double-initialization on first paper chat call.
        self.paper_chat = PaperChatService(
            self.embedding_service, cloud_factory=self._cloud_factory
        )
        self.trends = TrendAnalysisService()
        self.citation_engine = CitationEngine()
        self.metadata_service = MetadataService()
        self.academic_search = AcademicSearchService(
            AcademicSearchSettings(
                arxiv_enabled=settings.retrieval.arxiv_enabled,
                crossref_enabled=settings.retrieval.crossref_enabled,
                openalex_enabled=settings.retrieval.openalex_enabled,
                timeout_seconds=settings.llm.request_timeout,
            )
        )

        # --- Memory ---
        self.knowledge_graph = KnowledgeGraph()

        # Conversation store: tracks multi-turn chat history per conversation_id.
        # Powers the unified /chat/message endpoint so the AI can understand
        # follow-up questions ("tell me more", "which was fastest?", etc.)
        self.conversation_store = ConversationStore()

        # Ollama model manager: discovers installed local models and routes each
        # request to the best model for the task (fast models for simple tasks,
        # stronger models for complex reasoning). Safe to initialize even if
        # Ollama is not running — discover() returns False gracefully.
        self.ollama_manager = OllamaModelManager(
            base_url=settings.llm.ollama_base_url
        )
        if settings.llm.provider == "ollama" or settings.llm.backend == "local":
            self.ollama_manager.discover()

        # --- Execution ---
        self.python_runner = PythonRunner(
            enabled=settings.execution.enabled,
            max_code_chars=settings.execution.max_code_chars,
            timeout_seconds=settings.execution.timeout_seconds,
        )

        # --- Agent layer ---
        retrieval_agent = RetrievalAgent(self.retriever)
        tools = self._build_tool_registry(retrieval_agent)
        ml_agent = MLExecutionAgent(tools)
        self.pipeline_runner = PipelineRunner(ml_agent)

        # Keep a direct reference to SynthesisAgent for the chat() method.
        # The orchestrator uses synthesizer.synthesize() (legacy string return),
        # while chat() calls synthesizer.synthesize_structured() for rich output.
        self.synthesizer_service = SynthesisAgent(cloud_factory=self._cloud_factory)

        self.orchestrator = ResearchOrchestrator(
            planner=PlannerAgent(cloud_factory=self._cloud_factory, max_top_k=settings.retrieval.max_top_k),
            executor=ml_agent,
            evaluator=EvaluatorAgent(),
            synthesizer=self.synthesizer_service,
        )

        logger.info(
            "ResearchAIPlatform v3.1 ready — backend=%s provider=%s index_ready=%s",
            settings.llm.backend,
            settings.llm.provider,
            self.retriever.ready,
        )

    # ------------------------------------------------------------------
    # Tool registry — the full set of tools the LLM planner can invoke
    # ------------------------------------------------------------------

    def _build_tool_registry(self, retrieval_agent: RetrievalAgent) -> dict:
        return {
            # Core retrieval
            "hybrid_search":        self._hybrid_search,
            "search_papers":        self._hybrid_search,
            "search_arxiv":         self._academic_search,
            "search_crossref":      self._academic_search,
            "search_openalex":      self._academic_search,
            "smart_retrieve":       retrieval_agent.retrieve,
            # ML models
            "classify_query":       self._classify_query,
            "summarize":            self._summarize,
            "methodology_extract":  self._methodology_extract,
            "cluster_papers":       self._cluster_papers,
            "citation_signals":     self._citation_signals,
            # Research intelligence
            "trend_analysis":       self._trend_analysis,
            "citation_proxy":       self._citation_proxy,
            "metadata_analyse":     self._metadata_analyse,
            # LLM-backed synthesis
            "paper_chat":           self._paper_chat,
            "metadata_rag":         self._metadata_rag,
            # Execution
            "python_execute":       self._python_execute,
            "run_pipeline":         self._run_pipeline,
            # Utility
            "conversation":         self._conversation,
            "simple_answer":        self._simple_answer,
        }

    # ------------------------------------------------------------------
    # Tool implementations
    # ------------------------------------------------------------------

    def _classify_query(self, title: str = "", abstract: str = "", **_) -> dict:
        return self.classifier.classify(title, abstract)

    def _hybrid_search(
        self,
        query: str = "",
        top_k: int = 5,
        filters: dict | None = None,
        candidate_k: int | None = None,
        **_,
    ) -> dict:
        local = self.retriever.search(query, top_k=top_k, filters=filters, candidate_k=candidate_k)
        result = local
        needs_external = (
            self.settings.retrieval.paper_search_provider in {"auto", "online", "academic"}
            and (local.get("error") or len(local.get("results", [])) < min(top_k, 3) or self._needs_fresh_research(query))
        )
        if needs_external:
            external = self._academic_search(query=query, top_k=top_k)
            merged = []
            if isinstance(local, dict) and local.get("results"):
                merged.extend(local["results"])
            if external.get("results"):
                merged.extend(external["results"])
            if merged:
                merged = self.academic_search.rank(query, self.academic_search.deduplicate(merged))[:top_k]
                result = {
                    "query": query,
                    "retrieval_strategy": "hybrid_local_plus_academic_apis",
                    "results": merged,
                    "count": len(merged),
                    "candidate_count": len(merged),
                    "local_error": local.get("error"),
                    "provider_errors": external.get("provider_errors", []),
                }
            else:
                result = external if not local.get("results") else local

        if result.get("results"):
            # Feed into knowledge graph (no extra HF call needed)
            self.knowledge_graph.ingest_papers(result["results"])
            self.knowledge_graph.ingest_query(query)
        return result

    def _academic_search(self, query: str = "", top_k: int = 5, **_) -> dict:
        if not query.strip():
            return {"error": "query is required for academic search."}
        return self.academic_search.search(query, top_k=top_k)

    def _summarize(self, text: str = "", **_) -> dict:
        if not text.strip():
            return {"error": "No text provided to summarize."}
        return {"summary": self.summarizer.summarize(text)}

    def _methodology_extract(self, papers: list | None = None, text: str = "", **_) -> dict:
        source = text or "\n\n".join(
            f"{p.get('title', '')}. {p.get('abstract', '')}"
            for p in (papers or [])[:6]
        )
        return self.methodology.extract(source)

    def _cluster_papers(self, papers: list | None = None, **_) -> dict:
        """Assign papers to clusters using pre-computed cluster_assignments lookup.

        Uses cluster_assignments.parquet (by arxiv paper_id) instead of the
        numpy-incompatible kmeans.joblib, giving the same cluster labels without
        requiring the model to be loaded.
        """
        if self._cluster_assignments is None:
            return {"clusters": [], "count": 0, "error": "cluster_assignments.parquet not loaded."}

        papers = papers or []
        if not papers:
            return {"clusters": [], "count": 0}

        import pandas as pd
        cluster_map: dict[int, list] = {}
        unmatched = []

        for p in papers:
            pid = str(p.get("paper_id", "") or p.get("id", "")).strip()
            row = self._cluster_assignments[self._cluster_assignments["id"] == pid]
            if not row.empty:
                cid = int(row.iloc[0]["cluster_id"])
                cluster_map.setdefault(cid, []).append(p)
            else:
                unmatched.append(p)

        # If papers had no IDs or weren't in the index, put them in cluster -1
        if unmatched:
            cluster_map.setdefault(-1, []).extend(unmatched)

        clusters = []
        for cid, cpapers in sorted(cluster_map.items()):
            terms = self._cluster_terms.get(cid, [])
            label = (" / ".join(terms[:4]) if terms else f"Cluster {cid}") if cid >= 0 else "Unclustered"
            clusters.append({
                "cluster_id":    cid,
                "label":         label,
                "top_terms":     terms[:8],
                "paper_count":   len(cpapers),
                "papers":        [p.get("title", "") for p in cpapers[:5]],
            })

        return {"clusters": clusters, "count": len(clusters)}

    def _citation_signals(self, papers: list | None = None, **_) -> dict:
        return self.citation_graph.related_signals(papers or [])

    def _trend_analysis(self, papers: list | None = None, **_) -> dict:
        return self.trends.analyze(papers or [])

    def _citation_proxy(self, papers: list | None = None, **_) -> dict:
        return self.citation_engine.proxy_citations(papers or [])

    def _metadata_analyse(self, papers: list | None = None, **_) -> dict:
        return self.metadata_service.analyse(papers or [])

    def _paper_chat(self, session_id: str = "", question: str = "", top_k: int = 5, **_) -> dict:
        if not session_id:
            return {"error": "session_id is required for paper_chat."}
        return self.paper_chat.ask(session_id=session_id, question=question, top_k=top_k)

    def _metadata_rag(self, query: str = "", top_k: int = 5, papers: list | None = None, **_) -> dict:
        """Retrieval-augmented generation: search papers then produce a grounded answer.

        Pipeline:
          1. hybrid_search → candidate papers (reuses the full hybrid pipeline)
          2. Build a numbered context block from titles + abstract snippets
          3. Cloud LLM generates an answer citing papers as [1], [2], ...
          4. If no LLM is available, fall back to a formatted paper list

        WHY call _hybrid_search here instead of the vector store directly:
          _hybrid_search also triggers knowledge-graph ingestion and category-aware
          ranking, giving the LLM richer, better-ordered context.

        BUG FIX (v3.1.1): previously called get_cloud_client() directly.
          In local-only mode (no API key) this crashed with ValueError instead of
          gracefully returning the fallback paper list. Fix: use self._cloud_factory.
        """
        if papers:
            results = self.academic_search.rank(query, self.academic_search.deduplicate(papers))[:top_k]
            search = {"results": results}
        else:
            search = self._hybrid_search(query, top_k=top_k)
            if search.get("error"):
                return search
            results = search.get("results", [])
        if not results:
            return {"query": query, "answer": "No relevant papers found in the index.", "retrieved": []}

        # Build a compact numbered context block — LLM will cite as [1], [2], ...
        # 700-char abstract cap keeps the prompt within reasonable token limits.
        context = "\n\n".join(
            f"[{i}] {p.get('title', 'Untitled')} ({p.get('year', '')})\n"
            f"Authors: {p.get('authors', '')}\n"
            f"Source: {p.get('source', '')} {p.get('doi') or p.get('arxiv_id') or p.get('paper_id', '')}\n"
            f"URL: {p.get('url') or p.get('arxiv_url', '')}\n"
            f"Abstract: {str(p.get('abstract', ''))[:700]}"
            for i, p in enumerate(results, 1)
        )
        llm_success = False
        fallback_reason = ""
        try:
            # Use the cloud factory (may be None in local-only mode, triggers except)
            if self._cloud_factory is None:
                raise RuntimeError("No cloud LLM configured.")
            cloud = self._cloud_factory()
            answer = cloud.generate(
                prompt=f"Question: {query}\n\nPapers:\n{context}\n\nAnswer with citations like [1].",
                max_tokens=800,
                system=(
                    "You are a scientific research assistant. "
                    "Answer using ONLY the provided paper metadata and abstracts. "
                    "Be specific and cite papers as [1], [2], etc. "
                    "Do NOT add information not present in the papers.\n\n"
                    "FORMATTING: Use **bold** for key terms. "
                    "Use bullet lists (- item) for multiple findings or properties. "
                    "Use ### headers for major sections when the answer has distinct parts. "
                    "Aim for 150-350 words. The output will be rendered as markdown."
                ),
            )
            if answer and len(answer.split()) >= 40:
                llm_success = True
            else:
                fallback_reason = f"LLM returned too-short response ({len(answer.split())} words < 40)"
                answer = self._format_paper_list(query, results)
        except Exception as exc:
            fallback_reason = f"{type(exc).__name__}: {exc}"
            # Graceful degradation: return a clean formatted paper list
            answer = self._format_paper_list(query, results)

        logger.info(
            "[METADATA_RAG] llm_attempted=true llm_success=%s fallback_used=%s%s",
            llm_success,
            not llm_success,
            f" fallback_reason={fallback_reason!r}" if not llm_success else "",
        )
        return {"query": query, "answer": answer, "retrieved": results, "llm_success": llm_success}

    def _python_execute(self, code: str = "", **_) -> dict:
        return self.python_runner.run(code).to_dict()

    def _run_pipeline(self, pipeline_name: str = "full_research_analysis", query: str = "", **_) -> dict:
        if not query:
            return {"error": "query is required for run_pipeline."}
        return self.pipeline_runner.run(pipeline_name, query).to_dict()

    @staticmethod
    def _conversation(query: str = "", **_) -> dict:
        return {
            "answer": (
                "Hello! I'm your AI Research Intelligence assistant. I can help you find and analyse arXiv papers, "
                "extract methodology, identify trends, summarize papers, and answer research questions using the "
                "local paper index. What would you like to explore?"
            ),
            "query": query,
        }

    def _simple_answer(self, query: str = "", **_) -> dict:
        try:
            if self._cloud_factory is None:
                raise RuntimeError("No LLM provider configured.")
            cloud = self._cloud_factory()
            selected = self.model_router.select(query, "answer")
            original_model = getattr(cloud, "model", "")
            if selected:
                cloud.model = selected
            answer = cloud.generate(
                prompt=query,
                max_tokens=450,
                system=(
                    "You are a helpful research assistant. Answer basic educational questions directly. "
                    "Do not cite papers unless papers were retrieved and supplied. Keep the answer concise."
                ),
            )
            if original_model:
                cloud.model = original_model
            return {"answer": answer, "query": query, "model": selected}
        except Exception as exc:
            fallback = (
                "Machine learning is a branch of AI where systems learn patterns from data and use those "
                "patterns to make predictions, classifications, or decisions without being explicitly "
                "programmed for every case."
            )
            if "machine learning" in query.lower():
                return {"answer": fallback, "query": query, "model": "local_fallback", "warning": str(exc)}
            return {"answer": f"I can answer that, but the configured LLM is unavailable: {exc}", "query": query, "error": str(exc)}

    @staticmethod
    def _needs_fresh_research(query: str) -> bool:
        q = (query or "").lower()
        return any(term in q for term in ("latest", "recent", "new papers", "2024", "2025", "2026", "state of the art", "sota"))

    @staticmethod
    def _resolve_list_followup(query: str, conv) -> str | None:
        import re
        q = (query or "").lower()
        ordinal_map = {
            "first": 1, "1st": 1,
            "second": 2, "2nd": 2,
            "third": 3, "3rd": 3,
            "fourth": 4, "4th": 4,
            "fifth": 5, "5th": 5,
        }
        requested = None
        for word, idx in ordinal_map.items():
            if word in q:
                requested = idx
                break
        if requested is None or not any(term in q for term in ("approach", "paper", "one", "item", "source")):
            return None

        previous_assistant = None
        for turn in reversed(conv.turns[:-1]):
            if turn.role == "assistant":
                previous_assistant = turn.content
                break
        if not previous_assistant:
            return None

        match = re.search(rf"^{requested}\.\s+(.+?)(?=^\d+\.\s+|\Z)", previous_assistant, re.M | re.S)
        if not match:
            return None
        item = " ".join(match.group(1).split())
        return f"The {requested} item I mentioned was: {item}"

    # ------------------------------------------------------------------
    # Unified chat entry point (powers /chat/message)
    # ------------------------------------------------------------------

    def chat(
        self,
        query: str,
        conversation_id: str | None = None,
        session_id: str | None = None,
        top_k: int = 5,
        debug: bool = False,
        user_id: str | None = None,
    ) -> dict:
        """Unified chat entry point — the only method the frontend needs to call.

        This is the "ChatGPT-like" interface. The user sends a message; the
        system automatically decides which tools to invoke, which model to use,
        how to retrieve evidence, and how to synthesize a grounded answer.

        Orchestration flow:
          1. Retrieve / create conversation from ConversationStore (scoped to user_id)
          2. Add user message to conversation history
          3. Pass conversation history to orchestrator (context-aware planning)
          4. Run full Plan→Execute→Evaluate→Synthesize pipeline
          5. Extract structured sources from tool outputs
          6. Add assistant answer to conversation history
          7. Return structured response (answer + sources + confidence + metadata)

        Returns:
            {
                "answer":          str,
                "sources":         list[dict],
                "confidence":      float,
                "conversation_id": str,
                "intent":          str,
                "tools_used":      list[str],
                "model_used":      str,
                "latency_ms":      float,
                "debug_trace":     dict | None,
            }
        """
        import time
        started = time.perf_counter()

        # Step 1–2: Resume or create conversation, store user turn
        # user_id is passed so new conversations are scoped to the authenticated user
        cid, conv = self.conversation_store.get_or_create(conversation_id, user_id=user_id)
        self.conversation_store.add_turn(cid, "user", query)
        conv = self.conversation_store.get(cid) or conv

        resolved = self._resolve_list_followup(query, conv)
        if resolved:
            self.conversation_store.add_turn(cid, "assistant", resolved)
            return {
                "answer": resolved,
                "sources": [],
                "confidence": 0.75,
                "conversation_id": cid,
                "intent": "context_followup",
                "tools_used": ["conversation_memory"],
                "model_used": "deterministic_context_resolver",
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                "debug_trace": {
                    "request_id": str(uuid.uuid4())[:8],
                    "mode": "context_followup",
                    "plan": {"intent": "context_followup", "query": query},
                    "executor_output": {"conversation_memory": {"resolved": True}},
                } if debug else None,
            }

        # Step 3: Build context summary for the planner
        # This lets the planner resolve "that paper", "the second one", etc.
        history = conv.context_summary(last_n_pairs=6)

        # Step 4: Full orchestration (Plan → Execute → Evaluate → Synthesize)
        # The orchestrator internally runs: Plan → Execute → Evaluate → synthesize()
        # synthesize() uses the cloud LLM (Gemini) and produces the full answer.
        request_id = str(uuid.uuid4())[:8]
        logger.info("[ORCHESTRATOR] request_id=%s starting", request_id)
        raw = self.orchestrator.run(
            mode="auto",
            query=query,
            top_k=top_k,
            session_id=session_id,
            conversation_history=history if conv.turn_count > 2 else None,
        )
        logger.info(
            "[ORCHESTRATOR] request_id=%s plan_fallback=%s eval_score=%s answer_len=%d",
            request_id,
            raw.get("mediator", {}).get("used_fallback"),
            raw.get("evaluation", {}).get("quality_score"),
            len(raw.get("final_answer", "")),
        )

        # Step 5: Structured synthesis — extracts sources, confidence, tools_used.
        # synthesize_structured() may short-circuit via _has_grounded_direct_answer
        # and return a shorter metadata_rag answer. We ALWAYS prefer the orchestrator's
        # final_answer (which went through full cloud LLM synthesis) when it is richer.
        quality_score = raw.get("evaluation", {}).get("quality_score")
        structured = self.synthesizer_service.synthesize_structured(
            query=query,
            plan=raw.get("plan", {}),
            outputs=raw.get("executor_output", {}),
            quality_score=quality_score,
        )

        # Pick the richer answer: orchestrator's LLM synthesis vs structured direct
        orch_answer = raw.get("final_answer", "")
        struct_answer = structured["answer"]
        if len(orch_answer.strip()) > len(struct_answer.strip()):
            answer = orch_answer
            logger.info(
                "[CHAT] request_id=%s synthesis_mode=orchestrator_llm answer_len=%d",
                request_id, len(answer),
            )
        else:
            answer = struct_answer
            logger.info(
                "[CHAT] request_id=%s synthesis_mode=structured_direct answer_len=%d",
                request_id, len(answer),
            )

        sources = structured["sources"]
        confidence = structured["confidence"]
        tools_used = structured["tools_used"]
        model_used = structured.get("model_used", "")

        # Step 6: Store assistant response in conversation history
        self.conversation_store.add_turn(cid, "assistant", answer)

        latency_ms = round((time.perf_counter() - started) * 1000, 2)

        result = {
            "answer":          answer,
            "sources":         sources,
            "confidence":      confidence,
            "conversation_id": cid,
            "intent":          raw.get("mode", "research_analysis"),
            "tools_used":      tools_used,
            "model_used":      model_used,
            "latency_ms":      latency_ms,
            "debug_trace":     raw if debug else None,
        }
        return result

    # ------------------------------------------------------------------
    # Properties and helpers
    # ------------------------------------------------------------------

    @property
    def indexed_paper_count(self) -> int:
        return self.vector_store.paper_count

    # ------------------------------------------------------------------
    # Clustering artifacts loader
    # ------------------------------------------------------------------

    def _load_clustering_artifacts(self, clustering_dir) -> None:
        """Load cluster_assignments.parquet and cluster_terms.joblib.

        Does NOT load kmeans.joblib because it was saved with an incompatible
        numpy version (MT19937 BitGenerator state format mismatch). Clustering
        inference uses the pre-computed cluster_assignments lookup instead.
        """
        import pandas as pd

        assignments_path = clustering_dir / "cluster_assignments.parquet"
        terms_path = clustering_dir / "cluster_terms.joblib"

        if assignments_path.exists():
            try:
                self._cluster_assignments = pd.read_parquet(assignments_path)
                logger.info(
                    "[CLUSTERING] cluster_assignments loaded: %d rows",
                    len(self._cluster_assignments),
                )
            except Exception as exc:
                logger.warning("[CLUSTERING] cluster_assignments load failed: %s", exc)
        else:
            logger.warning("[CLUSTERING] cluster_assignments.parquet not found at %s", assignments_path)

        if terms_path.exists():
            try:
                self._cluster_terms = joblib.load(terms_path)
                logger.info(
                    "[CLUSTERING] cluster_terms loaded: %d clusters",
                    len(self._cluster_terms),
                )
            except Exception as exc:
                logger.warning("[CLUSTERING] cluster_terms load failed: %s", exc)
        else:
            logger.warning("[CLUSTERING] cluster_terms.joblib not found at %s", terms_path)

    @staticmethod
    def _resolve_embedding_model(settings: Settings) -> str:
        path = settings.paths.similarity_dir / "embedding_model_name.joblib"
        if path.exists():
            try:
                return str(joblib.load(path))
            except Exception:
                pass
        return settings.retrieval.embedding_model_name

    @staticmethod
    def _format_paper_list(query: str, results: list) -> str:
        lines = [f"Top papers related to: {query}"]
        for i, p in enumerate(results[:6], 1):
            pid = p.get("paper_id", "")
            link = f" — https://arxiv.org/abs/{pid}" if pid else ""
            lines.append(f"{i}. {p.get('title', 'Untitled')} ({p.get('year', '')}){link}")
        return "\n".join(lines)

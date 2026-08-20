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

from research_ai.agents.evaluator_agent import EvaluatorAgent
from research_ai.agents.ml_execution_agent import MLExecutionAgent
from research_ai.agents.orchestrator import ResearchOrchestrator
from research_ai.agents.planner import PlannerAgent
from research_ai.agents.retrieval_agent import RetrievalAgent
from research_ai.agents.synthesis_agent import SynthesisAgent
from research_ai.configs.settings import Settings
from research_ai.execution.pipelines import PipelineRunner
from research_ai.execution.python_runner import PythonRunner
from research_ai.llm import get_cloud_client
from research_ai.memory.conversation_store import ConversationStore
from research_ai.memory.knowledge_graph import KnowledgeGraph
from research_ai.ml_models.citation_graph import CitationGraphService
from research_ai.ml_models.ranking import RankingService
from research_ai.ml_models.similarity import SimilarityService
import os
from research_ai.remote_ml import (
    RemoteClassifierService,
    RemoteHybridSearchService,
    RemoteMethodologyExtractor,
    RemoteScientificSummarizer,
    RemoteClusteringService
)
from research_ai.ollama_manager import OllamaModelManager
from research_ai.research.citation_engine import CitationEngine
from research_ai.research.metadata import MetadataService
from research_ai.research.paper_ingestion import PaperChatService
from research_ai.research.trend_analysis import TrendAnalysisService
from research_ai.retrieval.embeddings import EmbeddingService
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
        hf_space_id = os.environ.get("HF_SPACE_ID", "").strip()
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

        # --- Core infrastructure ---
        self.embedding_service = EmbeddingService(self._resolve_embedding_model(settings))
        self.retriever = RemoteHybridSearchService(hf_space_id)

        # --- ML models ---
        self.classifier = RemoteClassifierService(hf_space_id)
        self.summarizer = RemoteScientificSummarizer(hf_space_id)
        self.similarity = SimilarityService(self.embedding_service)
        self.methodology = RemoteMethodologyExtractor(hf_space_id)
        self.clustering = RemoteClusteringService(hf_space_id)
        self.ranking = RankingService()
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
        result = self.retriever.search(query, top_k=top_k, filters=filters, candidate_k=candidate_k)
        if result.get("results"):
            # Feed into knowledge graph (no extra HF call needed)
            self.knowledge_graph.ingest_papers(result["results"])
            self.knowledge_graph.ingest_query(query)
        return result

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
        return self.clustering.cluster_papers(papers or [])

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

    def _metadata_rag(self, query: str = "", top_k: int = 5, **_) -> dict:
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
        search = self._hybrid_search(query, top_k=top_k)
        if search.get("error"):
            return search
        results = search.get("results", [])
        if not results:
            return {"query": query, "answer": "No relevant papers found in the index.", "retrieved": []}

        # Build a compact numbered context block — LLM will cite as [1], [2], ...
        # 700-char abstract cap keeps the prompt within reasonable token limits.
        context = "\n\n".join(
            f"[{i}] {p.get('title', 'Untitled')} ({p.get('year', '')})\n{str(p.get('abstract', ''))[:700]}"
            for i, p in enumerate(results, 1)
        )
        try:
            # Use the cloud factory (may be None in local-only mode, triggers except)
            if self._cloud_factory is None:
                raise RuntimeError("No cloud LLM configured.")
            cloud = self._cloud_factory()
            answer = cloud.generate(
                prompt=f"Question: {query}\n\nPapers:\n{context}\n\nAnswer with citations like [1].",
                max_tokens=512,
                system=(
                    "You are a scientific research assistant. "
                    "Answer using ONLY the provided paper metadata and abstracts. "
                    "Be specific. Cite papers as [1], [2], etc. "
                    "Do NOT add information not present in the papers."
                ),
            )
        except Exception:
            # Graceful degradation: return a clean formatted paper list
            answer = self._format_paper_list(query, results)
        return {"query": query, "answer": answer, "retrieved": results}

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
                "Hello! I'm your AI Research Intelligence assistant. I can help you:\n"
                "• Find and analyse arXiv papers on any topic\n"
                "• Extract methodology and experimental details\n"
                "• Identify research trends and citation patterns\n"
                "• Summarise papers or abstracts\n"
                "• Answer research questions using the paper database\n\n"
                "What would you like to explore?"
            ),
            "query": query,
        }

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
    ) -> dict:
        """Unified chat entry point — the only method the frontend needs to call.

        This is the "ChatGPT-like" interface. The user sends a message; the
        system automatically decides which tools to invoke, which model to use,
        how to retrieve evidence, and how to synthesize a grounded answer.

        Orchestration flow:
          1. Retrieve / create conversation from ConversationStore
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
        cid, conv = self.conversation_store.get_or_create(conversation_id)
        conv.add("user", query)

        # Step 3: Build context summary for the planner
        # This lets the planner resolve "that paper", "the second one", etc.
        history = conv.context_summary(last_n_pairs=6)

        # Step 4: Full orchestration (Plan → Execute → Evaluate → Synthesize)
        raw = self.orchestrator.run(
            mode="auto",
            query=query,
            top_k=top_k,
            session_id=session_id,
            conversation_history=history if conv.turn_count > 2 else None,
        )

        # Step 5: Structured synthesis (sources + confidence + tools_used)
        # SynthesisAgent.synthesize_structured provides richer output than the
        # legacy synthesize() method which returns only a string.
        quality_score = raw.get("evaluation", {}).get("quality_score")
        structured = self.synthesizer_service.synthesize_structured(
            query=query,
            plan=raw.get("plan", {}),
            outputs=raw.get("executor_output", {}),
            quality_score=quality_score,
        )

        answer = structured["answer"]
        sources = structured["sources"]
        confidence = structured["confidence"]
        tools_used = structured["tools_used"]
        model_used = structured.get("model_used", "")

        # Step 6: Store assistant response in conversation history
        conv.add("assistant", answer)

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
        return 0  # Remote index handles this

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

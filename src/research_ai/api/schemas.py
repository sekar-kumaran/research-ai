"""Pydantic request/response schemas for the Research AI Platform API."""
from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class ClassifyRequest(BaseModel):
    title: str = Field(default="")
    abstract: str = Field(default="")

    @field_validator("title", "abstract", mode="before")
    @classmethod
    def coerce_str(cls, value):
        return str(value) if value is not None else ""


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=2)
    top_k: int = Field(default=5, ge=1, le=20)
    filters: dict = Field(default_factory=dict)


class SummarizeRequest(BaseModel):
    text: str = Field(..., min_length=5)


class AskRequest(BaseModel):
    query: str = Field(..., min_length=2)
    top_k: int = Field(default=5, ge=1, le=20)


class AgentRequest(BaseModel):
    query: str = Field(..., min_length=1)
    mode: str = Field(default="auto")
    title: str | None = None
    abstract: str | None = None
    top_k: int = Field(default=5, ge=1, le=20)
    text: str | None = None
    session_id: str | None = None


class MediatorMeta(BaseModel):
    reason: str = ""
    used_fallback: bool = False


class MediatedAgentResponse(BaseModel):
    request_id: str
    mode: str
    mediator: MediatorMeta
    executor_output: dict
    final_answer: str
    latency_ms: float


class ArxivLoadRequest(BaseModel):
    arxiv_id: str = Field(..., min_length=3)


class PaperChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)


class SimilarityRequest(BaseModel):
    text_a: str = Field(..., min_length=5)
    text_b: str = Field(..., min_length=5)


class BulkChatRequest(BaseModel):
    arxiv_ids: list[str] = Field(..., min_length=1)
    question: str = Field(default="")

    @field_validator("arxiv_ids")
    @classmethod
    def non_empty_ids(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip() for item in value if item.strip()]
        if not cleaned:
            raise ValueError("arxiv_ids must contain at least one non-empty ID.")
        return cleaned


class PythonExecutionRequest(BaseModel):
    code: str = Field(..., min_length=1, max_length=4000)


class MetadataAnalyseRequest(BaseModel):
    """Request body for /metadata/analyse — accepts a list of paper dicts."""
    papers: list[dict] = Field(..., min_length=1)


class CitationProxyRequest(BaseModel):
    """Request body for citation intelligence endpoints."""
    papers: list[dict] = Field(..., min_length=1)


class PipelineRequest(BaseModel):
    """Request body for /pipeline/run."""
    pipeline_name: str = Field(default="full_research_analysis")
    query: str = Field(..., min_length=2)
    extra_args: dict = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Unified chat endpoint schemas
# These power the ChatGPT-like /chat/message and /chat/stream endpoints.
# The user sends only a query (and optionally a conversation_id to continue
# a prior session). Everything else — retrieval, model selection, pipeline
# routing — is decided by the AI orchestrator automatically.
# ---------------------------------------------------------------------------

class ChatMessageRequest(BaseModel):
    """Request for the unified /chat/message endpoint.

    The user only needs to send their natural-language query. All tool
    selection, retrieval strategy, model choice, and synthesis is handled
    internally by the AI orchestrator.

    Fields:
        query           — The user's question or instruction (required)
        conversation_id — Continue an existing conversation (optional).
                          If absent or unknown, a new conversation is started
                          and the ID is returned in the response.
        session_id      — Paper chat session to query against (optional).
                          If provided, the orchestrator will include per-paper
                          context in its reasoning.
        top_k           — Max results to retrieve (optional, default 5).
        debug           — If True, include orchestration trace in response.
    """
    query: str = Field(..., min_length=1)
    conversation_id: str | None = Field(default=None)
    session_id: str | None = Field(default=None)
    top_k: int = Field(default=5, ge=1, le=20)
    debug: bool = Field(default=False)


class SourcePaper(BaseModel):
    """A single retrieved paper cited in the response."""
    title: str = Field(default="")
    paper_id: str = Field(default="")
    year: str = Field(default="")
    category: str = Field(default="")
    abstract_snippet: str = Field(default="")
    score: float = Field(default=0.0)
    arxiv_url: str = Field(default="")


class ChatMessageResponse(BaseModel):
    """Response from the unified /chat/message endpoint.

    The response bundles:
      - answer         : The AI's conversational response text
      - sources        : List of retrieved papers that grounded the answer
      - confidence     : How well-evidenced the answer is (0–1)
      - conversation_id: The session ID to send back in the next message
      - tools_used     : Which internal tools the orchestrator invoked
      - model_used     : Which Ollama/cloud model generated the synthesis
      - latency_ms     : End-to-end request latency
      - debug_trace    : Full orchestration trace (only if debug=True)
    """
    answer: str
    sources: list[SourcePaper] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    conversation_id: str
    intent: str = Field(default="research_analysis")
    tools_used: list[str] = Field(default_factory=list)
    model_used: str = Field(default="")
    latency_ms: float = Field(default=0.0)
    debug_trace: dict | None = Field(default=None)


class ModelInfo(BaseModel):
    """Information about a locally available Ollama model."""
    name: str
    tier: int
    tier_label: str
    size_gb: float = Field(default=0.0)


class ModelsListResponse(BaseModel):
    """Response from /models/list."""
    available: bool
    models: list[ModelInfo] = Field(default_factory=list)
    default_model: str = Field(default="")

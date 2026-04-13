from pydantic import BaseModel, Field


class ClassifyRequest(BaseModel):
    title: str = Field(..., min_length=3)
    abstract: str = Field(..., min_length=10)


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=3)
    top_k: int = Field(default=5, ge=1, le=20)


class SummarizeRequest(BaseModel):
    text: str = Field(..., min_length=30)


class AskRequest(BaseModel):
    query: str = Field(..., min_length=3)
    top_k: int = Field(default=5, ge=1, le=20)


class AgentRequest(BaseModel):
    query: str = Field(..., min_length=3)
    mode: str = Field(default="auto", pattern="^(auto|classify|search|summarize|ask)$")
    title: str | None = None
    abstract: str | None = None
    top_k: int = Field(default=5, ge=1, le=20)


class ArxivLoadRequest(BaseModel):
    arxiv_id: str = Field(..., min_length=3)


class PaperChatRequest(BaseModel):
    session_id: str = Field(..., min_length=8)
    question: str = Field(..., min_length=3)
    top_k: int = Field(default=5, ge=1, le=20)

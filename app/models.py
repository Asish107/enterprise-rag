"""Pydantic request/response schemas for the API."""
from __future__ import annotations

from pydantic import BaseModel, Field


class IngestResponse(BaseModel):
    document_id: str
    filename: str
    format: str
    num_chunks: int
    num_characters: int


class Citation(BaseModel):
    chunk_id: str
    document_id: str
    filename: str
    score: float = Field(..., description="Semantic similarity score (higher = more relevant).")
    snippet: str


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int | None = Field(None, ge=1, le=20)


class QueryResponse(BaseModel):
    question: str
    answer: str
    citations: list[Citation]
    grounded: bool = Field(..., description="True when the answer is supported by retrieved context.")
    latency_ms: float
    model: str


class EvalRequest(BaseModel):
    """A batch of question/expected-answer pairs to score the pipeline against."""

    samples: list["EvalSample"]
    top_k: int | None = Field(None, ge=1, le=20)


class EvalSample(BaseModel):
    question: str
    # Optional reference answer / expected keywords used for coverage checks.
    expected_keywords: list[str] = Field(default_factory=list)


class EvalReport(BaseModel):
    num_samples: int
    retrieval_relevance: float = Field(..., description="Mean top-1 similarity across queries.")
    citation_coverage: float = Field(..., description="Fraction of answers with >=1 citation.")
    hallucination_rate: float = Field(..., description="Fraction of answers not grounded in context.")
    avg_latency_ms: float
    estimated_cost_usd: float
    details: list["EvalDetail"]


class EvalDetail(BaseModel):
    question: str
    top_score: float
    num_citations: int
    grounded: bool
    keyword_coverage: float
    latency_ms: float


EvalRequest.model_rebuild()
EvalReport.model_rebuild()

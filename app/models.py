"""Pydantic request/response schemas for the API."""
from __future__ import annotations

from pydantic import BaseModel, Field


class IngestResponse(BaseModel):
    document_id: str
    filename: str
    format: str
    num_chunks: int
    num_characters: int
    deduplicated: bool = Field(
        False, description="True if identical content was already ingested (no-op)."
    )


class DocumentInfo(BaseModel):
    document_id: str
    filename: str
    format: str
    num_chunks: int
    num_characters: int
    ingested_at: str


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0


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
    confidence: float = Field(..., description="Best dense-retrieval similarity for the query.")
    latency_ms: float
    model: str
    usage: Usage = Field(default_factory=Usage)


class EvalSample(BaseModel):
    """A labeled QA example for evaluation."""

    question: str
    ground_truth: str = Field("", description="Reference answer.")
    # Key facts that a correct, faithful answer/context should contain.
    expected_keywords: list[str] = Field(default_factory=list)
    # Optional: filename that should be the source of the answer.
    expected_source: str | None = None


class EvalRequest(BaseModel):
    """A batch of labeled samples to score the pipeline against."""

    samples: list[EvalSample]
    top_k: int | None = Field(None, ge=1, le=20)


class EvalMetrics(BaseModel):
    # -- RAGAS-style quality signals (0..1) --------------------------------
    faithfulness: float = Field(..., description="Answer claims supported by retrieved context.")
    answer_relevancy: float = Field(..., description="Semantic alignment of answer to question.")
    answer_correctness: float = Field(..., description="Token-F1 overlap with the ground truth.")
    context_precision: float = Field(..., description="Fraction of retrieved chunks that are relevant.")
    context_recall: float = Field(..., description="Fraction of expected facts present in context.")
    # -- operational signals -----------------------------------------------
    retrieval_relevance: float = Field(..., description="Mean dense-retrieval confidence.")
    citation_coverage: float = Field(..., description="Fraction of answers with >=1 citation.")
    hallucination_rate: float = Field(..., description="Fraction of answers not grounded.")
    avg_latency_ms: float
    total_cost_usd: float = Field(..., description="Real cost from token usage across the run.")


class EvalDetail(BaseModel):
    question: str
    faithfulness: float
    answer_relevancy: float
    answer_correctness: float
    context_precision: float
    context_recall: float
    grounded: bool
    num_citations: int
    latency_ms: float
    cost_usd: float


class EvalReport(BaseModel):
    num_samples: int
    metrics: EvalMetrics
    details: list[EvalDetail]

"""Evaluation of five production RAG signals.

    1. Retrieval relevance  — mean top-1 similarity of retrieved context.
    2. Citation coverage    — fraction of answers backed by >=1 citation.
    3. Hallucination rate   — fraction of answers not grounded in context.
    4. Latency              — average end-to-end query latency (ms).
    5. Cost                 — estimated USD spend for the generation calls.

These are intentionally lightweight, reference-free-ish signals so the harness
can run in CI. When a sample supplies ``expected_keywords`` we also compute a
keyword-coverage proxy for answer quality.
"""
from __future__ import annotations

from statistics import mean

from app.models import EvalDetail, EvalReport, EvalSample

# Rough blended output price (USD per 1K tokens) for the default model. Used only
# to give an order-of-magnitude cost signal, not billing-grade accounting.
_USD_PER_1K_TOKENS = 0.015


def _keyword_coverage(answer: str, keywords: list[str]) -> float:
    if not keywords:
        return 1.0
    answer_lower = answer.lower()
    hits = sum(1 for kw in keywords if kw.lower() in answer_lower)
    return hits / len(keywords)


def evaluate(samples: list[EvalSample], run_query) -> EvalReport:
    """Run each sample through ``run_query`` and aggregate the five signals.

    ``run_query(question)`` must return the API ``QueryResponse``.
    """
    details: list[EvalDetail] = []
    top_scores: list[float] = []
    latencies: list[float] = []
    grounded_flags: list[bool] = []
    cited_flags: list[bool] = []
    total_tokens = 0

    for sample in samples:
        resp = run_query(sample.question)
        top_score = resp.citations[0].score if resp.citations else 0.0
        num_citations = len(resp.citations)
        coverage = _keyword_coverage(resp.answer, sample.expected_keywords)

        top_scores.append(top_score)
        latencies.append(resp.latency_ms)
        grounded_flags.append(resp.grounded)
        cited_flags.append(num_citations > 0)
        # crude token estimate: ~4 chars/token over answer + context snippets
        total_tokens += (len(resp.answer) + sum(len(c.snippet) for c in resp.citations)) // 4

        details.append(
            EvalDetail(
                question=sample.question,
                top_score=round(top_score, 4),
                num_citations=num_citations,
                grounded=resp.grounded,
                keyword_coverage=round(coverage, 4),
                latency_ms=round(resp.latency_ms, 2),
            )
        )

    n = len(samples) or 1
    return EvalReport(
        num_samples=len(samples),
        retrieval_relevance=round(mean(top_scores) if top_scores else 0.0, 4),
        citation_coverage=round(sum(cited_flags) / n, 4),
        hallucination_rate=round(sum(1 for g in grounded_flags if not g) / n, 4),
        avg_latency_ms=round(mean(latencies) if latencies else 0.0, 2),
        estimated_cost_usd=round((total_tokens / 1000) * _USD_PER_1K_TOKENS, 6),
        details=details,
    )

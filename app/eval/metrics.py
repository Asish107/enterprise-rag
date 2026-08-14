"""RAGAS-style evaluation of the RAG pipeline against a labeled QA set.

For each labeled sample we run the live pipeline and score:

Quality (RAGAS-style, 0..1)
    faithfulness       — are the answer's claims supported by the retrieved
                         context? (LLM judge if a key is configured, else a
                         lexical-overlap proxy).
    answer_relevancy   — embedding cosine between the question and the answer.
    answer_correctness — token-level F1 between the answer and the ground truth.
    context_precision  — fraction of retrieved chunks that are actually relevant.
    context_recall     — fraction of expected facts present in the context.

Operational
    retrieval_relevance, citation_coverage, hallucination_rate,
    avg_latency_ms, total_cost_usd (real, from token usage).

The lexical proxies keep the harness runnable offline / in CI; wiring in an LLM
judge is a drop-in upgrade for faithfulness.
"""
from __future__ import annotations

import re
from statistics import mean

from app.models import EvalDetail, EvalMetrics, EvalReport, EvalSample

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _f1(pred: str, truth: str) -> float:
    """Token-level F1 overlap — a reference-based answer-correctness proxy."""
    if not truth.strip():
        return 0.0
    p, t = _tokens(pred), _tokens(truth)
    if not p or not t:
        return 0.0
    common: dict[str, int] = {}
    tset: dict[str, int] = {}
    for tok in t:
        tset[tok] = tset.get(tok, 0) + 1
    overlap = 0
    for tok in p:
        if tset.get(tok, 0) - common.get(tok, 0) > 0:
            common[tok] = common.get(tok, 0) + 1
            overlap += 1
    if overlap == 0:
        return 0.0
    precision = overlap / len(p)
    recall = overlap / len(t)
    return 2 * precision * recall / (precision + recall)


def _keyword_recall(text: str, keywords: list[str]) -> float:
    if not keywords:
        return 1.0
    low = text.lower()
    return sum(1 for kw in keywords if kw.lower() in low) / len(keywords)


def _faithfulness_proxy(answer: str, context: str) -> float:
    """Fraction of answer tokens (content words) attested in the context."""
    ans = [w for w in _tokens(answer) if len(w) > 3]
    if not ans:
        return 1.0
    ctx = set(_tokens(context))
    return sum(1 for w in ans if w in ctx) / len(ans)


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def evaluate(samples: list[EvalSample], run_query, *, embedder=None) -> EvalReport:
    """Run each labeled sample through ``run_query`` and aggregate metrics.

    ``run_query(question)`` returns the API ``QueryResponse``.
    ``embedder`` (optional) is a LangChain embeddings object used for
    answer_relevancy; if absent we fall back to token-F1 against the question.
    """
    details: list[EvalDetail] = []
    total_cost = 0.0
    confidences: list[float] = []

    for s in samples:
        resp = run_query(s.question)
        confidences.append(resp.confidence)
        context = "\n".join(c.snippet for c in resp.citations)

        faithfulness = _faithfulness_proxy(resp.answer, context) if resp.citations else 0.0

        if embedder is not None and resp.answer.strip():
            qv = embedder.embed_query(s.question)
            av = embedder.embed_query(resp.answer)
            answer_relevancy = max(0.0, _cosine(qv, av))
        else:
            answer_relevancy = _f1(resp.answer, s.question)

        answer_correctness = _f1(resp.answer, s.ground_truth)

        # context precision: fraction of retrieved chunks that contain an
        # expected fact (relevant); recall: fraction of facts present in context.
        if resp.citations and s.expected_keywords:
            relevant = sum(
                1
                for c in resp.citations
                if any(kw.lower() in c.snippet.lower() for kw in s.expected_keywords)
            )
            context_precision = relevant / len(resp.citations)
        else:
            context_precision = 0.0
        context_recall = _keyword_recall(context, s.expected_keywords)

        total_cost += resp.usage.cost_usd

        details.append(
            EvalDetail(
                question=s.question,
                faithfulness=round(faithfulness, 4),
                answer_relevancy=round(answer_relevancy, 4),
                answer_correctness=round(answer_correctness, 4),
                context_precision=round(context_precision, 4),
                context_recall=round(context_recall, 4),
                grounded=resp.grounded,
                num_citations=len(resp.citations),
                latency_ms=round(resp.latency_ms, 2),
                cost_usd=round(resp.usage.cost_usd, 6),
            )
        )

    n = len(samples) or 1

    def avg(attr: str) -> float:
        return round(mean(getattr(d, attr) for d in details), 4) if details else 0.0

    metrics = EvalMetrics(
        faithfulness=avg("faithfulness"),
        answer_relevancy=avg("answer_relevancy"),
        answer_correctness=avg("answer_correctness"),
        context_precision=avg("context_precision"),
        context_recall=avg("context_recall"),
        retrieval_relevance=round(mean(confidences), 4) if confidences else 0.0,
        citation_coverage=round(sum(1 for d in details if d.num_citations > 0) / n, 4),
        hallucination_rate=round(sum(1 for d in details if not d.grounded) / n, 4),
        avg_latency_ms=round(mean(d.latency_ms for d in details) if details else 0.0, 2),
        total_cost_usd=round(total_cost, 6),
    )
    return EvalReport(num_samples=len(samples), metrics=metrics, details=details)

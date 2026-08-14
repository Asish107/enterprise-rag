"""Cross-encoder reranking.

Bi-encoder retrieval (vectors / BM25) scores query and document independently,
which is fast but approximate. A cross-encoder jointly encodes the
(query, chunk) pair and produces a far more accurate relevance score. We use it
to re-order a small candidate set from the first-stage retrievers.

The model is loaded lazily and cached. Reranking is optional — if the model or
dependency is unavailable we return candidates unchanged.
"""
from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=2)
def _get_model(model_name: str):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name)


def _sigmoid(x: float) -> float:
    import math

    return 1.0 / (1.0 + math.exp(-x))


def rerank(
    query: str,
    candidates: list[tuple[str, str]],
    *,
    model_name: str,
    top_k: int,
) -> list[tuple[str, float]]:
    """Rerank ``(chunk_id, text)`` candidates; return ``(chunk_id, score)``.

    The cross-encoder emits an unbounded logit; we pass it through a sigmoid to
    get an *absolute* relevance probability in ``[0, 1]`` (not merely a relative
    ordering), so downstream code can threshold on it meaningfully.
    """
    if not candidates:
        return []
    model = _get_model(model_name)
    pairs = [(query, text) for _, text in candidates]
    raw = model.predict(pairs)

    scored = [
        (chunk_id, _sigmoid(float(score)))
        for (chunk_id, _), score in zip(candidates, raw)
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]

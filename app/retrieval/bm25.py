"""BM25 keyword search over the chunk corpus.

Complements dense vector search: BM25 catches exact terms, IDs, and rare
tokens that embeddings often blur. Built in-memory from the corpus chunks; the
index is cheap to rebuild whenever the corpus changes.
"""
from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    def __init__(self, chunk_ids: list[str], texts: list[str]):
        self.chunk_ids = chunk_ids
        self._bm25 = BM25Okapi([_tokenize(t) for t in texts]) if texts else None

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """Return ``(chunk_id, score)`` pairs, highest BM25 score first."""
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(zip(self.chunk_ids, scores), key=lambda x: x[1], reverse=True)
        return [(cid, float(s)) for cid, s in ranked[:top_k] if s > 0]

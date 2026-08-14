"""Hybrid retriever: dense + sparse fusion, then cross-encoder reranking.

Pipeline per query:

    1. Dense search (FAISS)   -> ranked chunk ids
    2. Sparse search (BM25)   -> ranked chunk ids
    3. Reciprocal Rank Fusion -> a single fused candidate list
    4. Cross-encoder rerank   -> final top-k, scored by relevance

Also owns ingestion (with content-hash dedup) and document management,
delegating vector storage to ``VectorStore`` and bookkeeping to ``Corpus``.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass

from app.config import Settings

from .bm25 import BM25Index
from .corpus import ChunkRecord, Corpus, DocumentRecord
from .reranker import rerank
from .vectorstore import VectorStore

# RRF constant; 60 is the value from the original Cormack et al. paper.
RRF_K = 60


@dataclass
class RetrievedChunk:
    chunk_id: str
    document_id: str
    filename: str
    text: str
    score: float  # final relevance in [0, 1]


def _rrf_fuse(*ranked_lists: list[str]) -> dict[str, float]:
    """Reciprocal Rank Fusion over several ranked chunk-id lists."""
    fused: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, chunk_id in enumerate(ranked):
            fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank + 1)
    return fused


class Retriever:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.corpus = Corpus.load(settings.index_dir)
        self.vectors = VectorStore(settings.embedding_model, settings.index_dir)
        self._bm25: BM25Index | None = None
        self._by_id: dict[str, ChunkRecord] = {c.chunk_id: c for c in self.corpus.chunks}

    # -- index maintenance --------------------------------------------------
    def _rebuild_lexical(self) -> None:
        self._by_id = {c.chunk_id: c for c in self.corpus.chunks}
        self._bm25 = BM25Index(
            [c.chunk_id for c in self.corpus.chunks],
            [c.text for c in self.corpus.chunks],
        )

    @property
    def bm25(self) -> BM25Index:
        if self._bm25 is None:
            self._rebuild_lexical()
        return self._bm25  # type: ignore[return-value]

    # -- ingestion ----------------------------------------------------------
    def ingest(
        self,
        *,
        text: str,
        chunks: list[str],
        filename: str,
        fmt: str,
        ingested_at: str,
    ) -> tuple[DocumentRecord, bool]:
        """Ingest a document. Returns ``(record, is_new)``.

        De-duplicates on the SHA-256 of the document text: re-ingesting
        identical content returns the existing record with ``is_new=False``.
        """
        content_hash = Corpus.content_hash(text)
        existing = self.corpus.find_by_hash(content_hash)
        if existing is not None:
            return existing, False

        document_id = uuid.uuid4().hex[:12]
        chunk_records = [
            ChunkRecord(
                chunk_id=f"{document_id}:{i}",
                document_id=document_id,
                filename=filename,
                text=chunk,
            )
            for i, chunk in enumerate(chunks)
        ]
        record = DocumentRecord(
            document_id=document_id,
            filename=filename,
            format=fmt,
            content_hash=content_hash,
            num_chunks=len(chunk_records),
            num_characters=len(text),
            ingested_at=ingested_at,
        )

        self.vectors.add([c.chunk_id for c in chunk_records], [c.text for c in chunk_records])
        self.corpus.add_document(record, chunk_records)
        self.corpus.save()
        self._rebuild_lexical()
        return record, True

    # -- document management ------------------------------------------------
    def list_documents(self) -> list[DocumentRecord]:
        return list(self.corpus.documents.values())

    def delete_document(self, document_id: str) -> bool:
        chunk_ids = [c.chunk_id for c in self.corpus.chunks_for(document_id)]
        removed = self.corpus.remove_document(document_id)
        if removed:
            self.vectors.delete(chunk_ids)
            self.corpus.save()
            self._rebuild_lexical()
        return removed

    # -- retrieval ----------------------------------------------------------
    def retrieve(self, query: str, top_k: int) -> tuple[list[RetrievedChunk], float]:
        """Return ``(results, confidence)``.

        ``confidence`` is the best dense cosine similarity for the query — an
        absolute, model-agnostic signal used to decide whether the corpus
        actually contains relevant material (grounding / out-of-domain guard),
        independent of the reranker's per-candidate scores.
        """
        if not self.corpus.chunks:
            return [], 0.0

        # First stage: pull a wider candidate pool from each retriever.
        pool = max(top_k * 5, 20)
        dense_scored = self.vectors.search(query, pool)
        confidence = max((s for _, s in dense_scored), default=0.0)
        dense = [cid for cid, _ in dense_scored]
        sparse = [cid for cid, _ in self.bm25.search(query, pool)]

        fused = _rrf_fuse(dense, sparse)
        if not fused:
            return [], confidence
        candidate_ids = sorted(fused, key=fused.get, reverse=True)[:pool]

        # Second stage: cross-encoder rerank the fused candidates.
        candidates = [
            (cid, self._by_id[cid].text) for cid in candidate_ids if cid in self._by_id
        ]
        if self.settings.rerank_enabled:
            try:
                ranked = rerank(
                    query,
                    candidates,
                    model_name=self.settings.rerank_model,
                    top_k=top_k,
                )
            except Exception:  # noqa: BLE001 - fall back to fused order
                ranked = [(cid, fused[cid]) for cid, _ in candidates[:top_k]]
        else:
            ranked = [(cid, fused[cid]) for cid, _ in candidates[:top_k]]

        results: list[RetrievedChunk] = []
        for chunk_id, score in ranked:
            rec = self._by_id[chunk_id]
            results.append(
                RetrievedChunk(
                    chunk_id=rec.chunk_id,
                    document_id=rec.document_id,
                    filename=rec.filename,
                    text=rec.text,
                    score=round(float(score), 4),
                )
            )
        return results, confidence

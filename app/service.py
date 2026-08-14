"""RAG service — the orchestration layer that ties the pieces together.

Ingestion:  load -> chunk -> embed (FAISS) + register (corpus, dedup)
Query:      hybrid retrieve (BM25 + vector) -> rerank -> grounded generation -> cite
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

from app.config import Settings
from app.generation import generate_answer
from app.ingestion import chunk_text, load_document
from app.models import Citation, DocumentInfo, IngestResponse, QueryResponse, Usage
from app.retrieval import Retriever


class RAGService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.retriever = Retriever(settings)

    # -- ingestion ----------------------------------------------------------
    def ingest_file(self, path: Path | str, *, filename: str | None = None) -> IngestResponse:
        path = Path(path)
        filename = filename or path.name
        text, fmt = load_document(path)
        chunks = chunk_text(
            text,
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )
        record, is_new = self.retriever.ingest(
            text=text,
            chunks=chunks,
            filename=filename,
            fmt=fmt,
            ingested_at=datetime.now(timezone.utc).isoformat(),
        )
        return IngestResponse(
            document_id=record.document_id,
            filename=record.filename,
            format=record.format,
            num_chunks=record.num_chunks,
            num_characters=record.num_characters,
            deduplicated=not is_new,
        )

    # -- document management ------------------------------------------------
    def list_documents(self) -> list[DocumentInfo]:
        return [
            DocumentInfo(
                document_id=r.document_id,
                filename=r.filename,
                format=r.format,
                num_chunks=r.num_chunks,
                num_characters=r.num_characters,
                ingested_at=r.ingested_at,
            )
            for r in self.retriever.list_documents()
        ]

    def delete_document(self, document_id: str) -> bool:
        return self.retriever.delete_document(document_id)

    # -- query --------------------------------------------------------------
    def query(self, question: str, top_k: int | None = None) -> QueryResponse:
        start = time.perf_counter()
        k = top_k or self.settings.top_k
        chunks, confidence = self.retriever.retrieve(question, top_k=k)

        result = generate_answer(
            question,
            chunks,
            confidence=confidence,
            anthropic_api_key=self.settings.anthropic_api_key,
            openrouter_api_key=self.settings.openrouter_api_key,
            openrouter_base_url=self.settings.openrouter_base_url,
            model=self.settings.generation_model,
            max_tokens=self.settings.max_tokens,
        )

        citations = (
            [
                Citation(
                    chunk_id=c.chunk_id,
                    document_id=c.document_id,
                    filename=c.filename,
                    score=c.score,
                    snippet=(c.text[:280] + "...") if len(c.text) > 280 else c.text,
                )
                for c in chunks
            ]
            if result.grounded
            else []
        )

        latency_ms = (time.perf_counter() - start) * 1000.0
        return QueryResponse(
            question=question,
            answer=result.answer,
            citations=citations,
            grounded=result.grounded,
            confidence=round(confidence, 4),
            latency_ms=round(latency_ms, 2),
            model=result.model,
            usage=Usage(
                prompt_tokens=result.usage.prompt_tokens,
                completion_tokens=result.usage.completion_tokens,
                total_tokens=result.usage.total_tokens,
                cost_usd=result.usage.cost_usd,
            ),
        )

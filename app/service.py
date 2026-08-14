"""RAG service — the orchestration layer that ties the pieces together.

Ingestion:  load -> chunk -> embed -> FAISS
Query:      embed question -> semantic search -> grounded generation -> cite
"""
from __future__ import annotations

import time
import uuid
from pathlib import Path

from app.config import Settings
from app.generation import generate_answer
from app.ingestion import chunk_text, load_document
from app.models import Citation, IngestResponse, QueryResponse
from app.retrieval import VectorStore


class RAGService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.store = VectorStore(settings.embedding_model, settings.index_dir)

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
        document_id = uuid.uuid4().hex[:12]
        num = self.store.add_chunks(chunks, document_id=document_id, filename=filename)
        return IngestResponse(
            document_id=document_id,
            filename=filename,
            format=fmt,
            num_chunks=num,
            num_characters=len(text),
        )

    # -- query --------------------------------------------------------------
    def query(self, question: str, top_k: int | None = None) -> QueryResponse:
        start = time.perf_counter()
        k = top_k or self.settings.top_k
        chunks = self.store.search(question, top_k=k)

        result = generate_answer(
            question,
            chunks,
            api_key=self.settings.anthropic_api_key,
            model=self.settings.generation_model,
            max_tokens=self.settings.max_tokens,
        )

        citations = [
            Citation(
                chunk_id=c.chunk_id,
                document_id=c.document_id,
                filename=c.filename,
                score=c.score,
                snippet=(c.text[:280] + "...") if len(c.text) > 280 else c.text,
            )
            for c in chunks
        ] if result.grounded else []

        latency_ms = (time.perf_counter() - start) * 1000.0
        return QueryResponse(
            question=question,
            answer=result.answer,
            citations=citations,
            grounded=result.grounded,
            latency_ms=round(latency_ms, 2),
            model=result.model,
        )

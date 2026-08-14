"""FAISS-backed vector store wrapper.

Wraps ``langchain_community.vectorstores.FAISS`` with a small, task-focused
API: add document chunks, run semantic search, and persist/load the index
from disk. Similarity scores are normalized to a ``[0, 1]`` relevance where
higher means more relevant, which keeps the eval signals intuitive.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document

from .embeddings import get_embeddings


@dataclass
class RetrievedChunk:
    chunk_id: str
    document_id: str
    filename: str
    text: str
    score: float  # 0..1, higher is more relevant


class VectorStore:
    def __init__(self, embedding_model: str, index_dir: Path):
        self.embedding_model = embedding_model
        self.index_dir = Path(index_dir)
        self._embeddings = get_embeddings(embedding_model)
        self._store = self._load_or_none()

    # -- persistence --------------------------------------------------------
    def _load_or_none(self):
        from langchain_community.vectorstores import FAISS

        faiss_file = self.index_dir / "index.faiss"
        if faiss_file.exists():
            return FAISS.load_local(
                str(self.index_dir),
                self._embeddings,
                allow_dangerous_deserialization=True,
            )
        return None

    def _persist(self) -> None:
        if self._store is not None:
            self.index_dir.mkdir(parents=True, exist_ok=True)
            self._store.save_local(str(self.index_dir))

    # -- writes -------------------------------------------------------------
    def add_chunks(self, chunks: list[str], *, document_id: str, filename: str) -> int:
        from langchain_community.vectorstores import FAISS

        docs = [
            Document(
                page_content=chunk,
                metadata={
                    "chunk_id": f"{document_id}:{i}",
                    "document_id": document_id,
                    "filename": filename,
                },
            )
            for i, chunk in enumerate(chunks)
        ]
        if not docs:
            return 0
        if self._store is None:
            self._store = FAISS.from_documents(docs, self._embeddings)
        else:
            self._store.add_documents(docs)
        self._persist()
        return len(docs)

    # -- reads --------------------------------------------------------------
    def search(self, query: str, top_k: int = 4) -> list[RetrievedChunk]:
        if self._store is None:
            return []
        # FAISS returns squared L2 distance on normalized vectors. For unit
        # vectors, ||a-b||^2 = 2 - 2*cos, so cosine = 1 - distance/2. We clamp
        # to [0, 1] so the score reads as an intuitive relevance value.
        results = self._store.similarity_search_with_score(query, k=top_k)
        retrieved: list[RetrievedChunk] = []
        for doc, distance in results:
            cosine = 1.0 - float(distance) / 2.0
            similarity = max(0.0, min(1.0, cosine))
            retrieved.append(
                RetrievedChunk(
                    chunk_id=doc.metadata.get("chunk_id", str(uuid.uuid4())),
                    document_id=doc.metadata.get("document_id", "unknown"),
                    filename=doc.metadata.get("filename", "unknown"),
                    text=doc.page_content,
                    score=round(similarity, 4),
                )
            )
        return retrieved

    @property
    def is_empty(self) -> bool:
        return self._store is None

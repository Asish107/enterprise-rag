"""FAISS-backed dense vector store (thin wrapper).

Owns only the embedding index. Document/chunk bookkeeping lives in ``Corpus``
and fusion/reranking in ``Retriever``; this class just embeds, persists, and
returns scored chunk ids for a query.
"""
from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document

from .embeddings import get_embeddings


class VectorStore:
    def __init__(self, embedding_model: str, index_dir: Path):
        self.embedding_model = embedding_model
        self.index_dir = Path(index_dir)
        self._embeddings = get_embeddings(embedding_model)
        self._store = self._load_or_none()

    # -- persistence --------------------------------------------------------
    def _load_or_none(self):
        from langchain_community.vectorstores import FAISS

        if (self.index_dir / "index.faiss").exists():
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
    def add(self, chunk_ids: list[str], texts: list[str]) -> None:
        from langchain_community.vectorstores import FAISS

        docs = [
            Document(page_content=text, metadata={"chunk_id": cid})
            for cid, text in zip(chunk_ids, texts)
        ]
        if not docs:
            return
        if self._store is None:
            self._store = FAISS.from_documents(docs, self._embeddings)
        else:
            self._store.add_documents(docs)
        self._persist()

    def delete(self, chunk_ids: list[str]) -> None:
        if self._store is None or not chunk_ids:
            return
        # Map external chunk_id metadata to FAISS internal docstore ids.
        internal = [
            fid
            for fid, doc in self._store.docstore._dict.items()
            if doc.metadata.get("chunk_id") in set(chunk_ids)
        ]
        if internal:
            self._store.delete(internal)
            self._persist()

    # -- reads --------------------------------------------------------------
    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """Return ``(chunk_id, cosine_similarity)`` pairs."""
        if self._store is None:
            return []
        results = self._store.similarity_search_with_score(query, k=top_k)
        out: list[tuple[str, float]] = []
        for doc, distance in results:
            # squared-L2 on unit vectors -> cosine = 1 - d/2
            cosine = max(0.0, min(1.0, 1.0 - float(distance) / 2.0))
            out.append((doc.metadata["chunk_id"], cosine))
        return out

    @property
    def is_empty(self) -> bool:
        return self._store is None

"""Persistent corpus: the document registry and the chunk store.

This is the source of truth for *what* has been ingested (independent of the
FAISS vectors). It enables:

* de-duplication — a document is identified by the SHA-256 of its text, so
  re-ingesting the same content is a no-op;
* a ``/documents`` listing and deletion;
* BM25 keyword search, which needs the raw chunk texts in memory.

Persisted as two small JSON files next to the FAISS index.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class DocumentRecord:
    document_id: str
    filename: str
    format: str
    content_hash: str
    num_chunks: int
    num_characters: int
    ingested_at: str


@dataclass
class ChunkRecord:
    chunk_id: str
    document_id: str
    filename: str
    text: str


@dataclass
class Corpus:
    index_dir: Path
    documents: dict[str, DocumentRecord] = field(default_factory=dict)
    chunks: list[ChunkRecord] = field(default_factory=list)

    # -- paths --------------------------------------------------------------
    @property
    def _docs_path(self) -> Path:
        return self.index_dir / "documents.json"

    @property
    def _chunks_path(self) -> Path:
        return self.index_dir / "chunks.json"

    # -- lifecycle ----------------------------------------------------------
    @classmethod
    def load(cls, index_dir: Path) -> "Corpus":
        index_dir = Path(index_dir)
        corpus = cls(index_dir=index_dir)
        if corpus._docs_path.exists():
            raw = json.loads(corpus._docs_path.read_text())
            corpus.documents = {k: DocumentRecord(**v) for k, v in raw.items()}
        if corpus._chunks_path.exists():
            raw = json.loads(corpus._chunks_path.read_text())
            corpus.chunks = [ChunkRecord(**c) for c in raw]
        return corpus

    def save(self) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self._docs_path.write_text(
            json.dumps({k: asdict(v) for k, v in self.documents.items()}, indent=2)
        )
        self._chunks_path.write_text(
            json.dumps([asdict(c) for c in self.chunks], indent=2)
        )

    # -- helpers ------------------------------------------------------------
    @staticmethod
    def content_hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def find_by_hash(self, content_hash: str) -> DocumentRecord | None:
        for rec in self.documents.values():
            if rec.content_hash == content_hash:
                return rec
        return None

    def chunks_for(self, document_id: str) -> list[ChunkRecord]:
        return [c for c in self.chunks if c.document_id == document_id]

    def add_document(self, record: DocumentRecord, chunks: list[ChunkRecord]) -> None:
        self.documents[record.document_id] = record
        self.chunks.extend(chunks)

    def remove_document(self, document_id: str) -> bool:
        if document_id not in self.documents:
            return False
        del self.documents[document_id]
        self.chunks = [c for c in self.chunks if c.document_id != document_id]
        return True

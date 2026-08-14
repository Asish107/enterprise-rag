"""Text chunking built on LangChain's recursive character splitter."""
from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_text(text: str, chunk_size: int = 800, chunk_overlap: int = 120) -> list[str]:
    """Split ``text`` into overlapping chunks suitable for embedding.

    Overlap preserves context across boundaries so retrieval doesn't clip
    a sentence that straddles two chunks.
    """
    if not text.strip():
        return []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    return [c.strip() for c in splitter.split_text(text) if c.strip()]

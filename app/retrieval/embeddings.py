"""Embedding backend.

Uses a local sentence-transformers model via LangChain's
``HuggingFaceEmbeddings`` so semantic search works with no external API and no
per-token cost. The model is loaded lazily and cached process-wide.
"""
from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=4)
def get_embeddings(model_name: str):
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=model_name,
        encode_kwargs={"normalize_embeddings": True},
    )

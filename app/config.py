"""Application configuration, loaded from environment variables."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for the RAG service.

    Values are read from environment variables (or a local ``.env`` file).
    Sensible defaults let the service boot with zero configuration.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Storage -----------------------------------------------------------
    data_dir: Path = Path(os.getenv("RAG_DATA_DIR", "data"))
    index_dir: Path = Path(os.getenv("RAG_INDEX_DIR", "data/index"))

    # --- Chunking ----------------------------------------------------------
    chunk_size: int = int(os.getenv("RAG_CHUNK_SIZE", "800"))
    chunk_overlap: int = int(os.getenv("RAG_CHUNK_OVERLAP", "120"))

    # --- Embeddings --------------------------------------------------------
    # A small, fast, fully-local sentence-transformers model. No API key needed.
    embedding_model: str = os.getenv("RAG_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    # --- Retrieval ---------------------------------------------------------
    top_k: int = int(os.getenv("RAG_TOP_K", "4"))

    # --- Generation --------------------------------------------------------
    # If an Anthropic API key is present we use Claude for grounded generation,
    # otherwise we fall back to a deterministic extractive answer so the whole
    # pipeline stays runnable offline / in CI.
    anthropic_api_key: str | None = os.getenv("ANTHROPIC_API_KEY")
    generation_model: str = os.getenv("RAG_GENERATION_MODEL", "claude-sonnet-5")
    max_tokens: int = int(os.getenv("RAG_MAX_TOKENS", "1024"))

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_dirs()
    return settings

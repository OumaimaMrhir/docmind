"""Configuration de l'application, lue depuis les variables d'environnement."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


@dataclass(frozen=True)
class Settings:
    # Fournisseur LLM : "openai", "anthropic" ou "extractive" (sans clé, pour la démo et les tests)
    llm_provider: str = field(default_factory=lambda: _env("LLM_PROVIDER", "extractive"))
    openai_api_key: str = field(default_factory=lambda: _env("OPENAI_API_KEY"))
    openai_model: str = field(default_factory=lambda: _env("OPENAI_MODEL", "gpt-4o-mini"))
    anthropic_api_key: str = field(default_factory=lambda: _env("ANTHROPIC_API_KEY"))
    anthropic_model: str = field(default_factory=lambda: _env("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001"))

    # Embeddings : "openai" ou "hashing" (local, sans clé)
    embedding_provider: str = field(default_factory=lambda: _env("EMBEDDING_PROVIDER", "hashing"))
    openai_embedding_model: str = field(default_factory=lambda: _env("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))

    # Stockage vectoriel : "memory" ou "pgvector"
    vector_store: str = field(default_factory=lambda: _env("VECTOR_STORE", "memory"))
    database_url: str = field(default_factory=lambda: _env("DATABASE_URL", "postgresql://docmind:docmind@localhost:5432/docmind"))

    # Découpage et recherche
    chunk_size: int = field(default_factory=lambda: int(_env("CHUNK_SIZE", "800")))
    chunk_overlap: int = field(default_factory=lambda: int(_env("CHUNK_OVERLAP", "120")))
    top_k: int = field(default_factory=lambda: int(_env("TOP_K", "4")))

    # Sécurité : si API_KEY est définie, chaque requête doit envoyer l'en-tête X-API-Key
    api_key: str = field(default_factory=lambda: _env("API_KEY"))
    cors_origins: str = field(default_factory=lambda: _env("CORS_ORIGINS", "http://localhost:5173"))

    @property
    def embedding_dim(self) -> int:
        return 1536 if self.embedding_provider == "openai" else 384


def get_settings() -> Settings:
    return Settings()

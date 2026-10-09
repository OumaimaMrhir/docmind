"""Stockage vectoriel : en mémoire (tests, démo) ou PostgreSQL + pgvector (production)."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Protocol

from .embeddings import cosine


@dataclass
class Chunk:
    doc_id: str
    doc_name: str
    index: int
    content: str
    embedding: list[float]


@dataclass
class SearchResult:
    doc_id: str
    doc_name: str
    index: int
    content: str
    score: float


@dataclass
class DocumentInfo:
    doc_id: str
    doc_name: str
    chunks: int


class VectorStore(Protocol):
    def add(self, chunks: list[Chunk]) -> None: ...
    def search(self, embedding: list[float], k: int) -> list[SearchResult]: ...
    def list_documents(self) -> list[DocumentInfo]: ...
    def delete_document(self, doc_id: str) -> bool: ...


def new_doc_id() -> str:
    return uuid.uuid4().hex[:12]


class InMemoryStore:
    def __init__(self) -> None:
        self._chunks: list[Chunk] = []

    def add(self, chunks: list[Chunk]) -> None:
        self._chunks.extend(chunks)

    def search(self, embedding: list[float], k: int) -> list[SearchResult]:
        scored = [SearchResult(c.doc_id, c.doc_name, c.index, c.content, cosine(embedding, c.embedding)) for c in self._chunks]
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:k]

    def list_documents(self) -> list[DocumentInfo]:
        docs: dict[str, DocumentInfo] = {}
        for c in self._chunks:
            info = docs.setdefault(c.doc_id, DocumentInfo(c.doc_id, c.doc_name, 0))
            info.chunks += 1
        return list(docs.values())

    def delete_document(self, doc_id: str) -> bool:
        before = len(self._chunks)
        self._chunks = [c for c in self._chunks if c.doc_id != doc_id]
        return len(self._chunks) < before


class PgVectorStore:
    """Stockage dans PostgreSQL avec l'extension pgvector (distance cosinus `<=>`)."""

    def __init__(self, database_url: str, dim: int) -> None:
        import psycopg  # import local : seulement nécessaire en mode pgvector

        self._conn = psycopg.connect(database_url, autocommit=True)
        self.dim = dim
        with self._conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS chunks (
                    id BIGSERIAL PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    doc_name TEXT NOT NULL,
                    chunk_index INT NOT NULL,
                    content TEXT NOT NULL,
                    embedding vector({dim}) NOT NULL
                )
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS chunks_doc_id_idx ON chunks (doc_id)")

    @staticmethod
    def _vec(embedding: list[float]) -> str:
        return "[" + ",".join(f"{v:.6f}" for v in embedding) + "]"

    def add(self, chunks: list[Chunk]) -> None:
        with self._conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO chunks (doc_id, doc_name, chunk_index, content, embedding) VALUES (%s, %s, %s, %s, %s::vector)",
                [(c.doc_id, c.doc_name, c.index, c.content, self._vec(c.embedding)) for c in chunks],
            )

    def search(self, embedding: list[float], k: int) -> list[SearchResult]:
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT doc_id, doc_name, chunk_index, content, 1 - (embedding <=> %s::vector) AS score "
                "FROM chunks ORDER BY embedding <=> %s::vector LIMIT %s",
                (self._vec(embedding), self._vec(embedding), k),
            )
            return [SearchResult(*row) for row in cur.fetchall()]

    def list_documents(self) -> list[DocumentInfo]:
        with self._conn.cursor() as cur:
            cur.execute("SELECT doc_id, doc_name, COUNT(*) FROM chunks GROUP BY doc_id, doc_name ORDER BY doc_name")
            return [DocumentInfo(*row) for row in cur.fetchall()]

    def delete_document(self, doc_id: str) -> bool:
        with self._conn.cursor() as cur:
            cur.execute("DELETE FROM chunks WHERE doc_id = %s", (doc_id,))
            return cur.rowcount > 0

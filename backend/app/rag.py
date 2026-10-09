"""Service RAG : ingestion des documents et réponse aux questions avec sources."""

from __future__ import annotations

import time
from dataclasses import dataclass

from .chunking import chunk_text
from .config import Settings
from .embeddings import Embedder, HashingEmbedder, OpenAIEmbedder
from .llm import LLM, AnthropicLLM, ExtractiveLLM, OpenAILLM
from .store import Chunk, InMemoryStore, PgVectorStore, SearchResult, VectorStore, new_doc_id

MIN_SCORE = 0.05


@dataclass
class Answer:
    answer: str
    sources: list[SearchResult]
    provider: str
    latency_ms: int


class RagService:
    def __init__(self, embedder: Embedder, store: VectorStore, llm: LLM, settings: Settings) -> None:
        self.embedder, self.store, self.llm, self.settings = embedder, store, llm, settings

    def ingest(self, doc_name: str, text: str) -> tuple[str, int]:
        pieces = chunk_text(text, self.settings.chunk_size, self.settings.chunk_overlap)
        if not pieces:
            raise ValueError("Le document est vide.")
        doc_id = new_doc_id()
        vectors = self.embedder.embed(pieces)
        self.store.add(
            [Chunk(doc_id, doc_name, i, piece, vec) for i, (piece, vec) in enumerate(zip(pieces, vectors, strict=True))]
        )
        return doc_id, len(pieces)

    def retrieve(self, question: str, k: int | None = None) -> list[SearchResult]:
        [query_vec] = self.embedder.embed([question])
        return self.store.search(query_vec, k or self.settings.top_k)

    def ask(self, question: str, k: int | None = None) -> Answer:
        start = time.perf_counter()
        results = self.retrieve(question, k)
        # On ne garde que les extraits pertinents : proches du meilleur résultat et au-dessus d'un seuil minimal
        top = results[0].score if results else 0.0
        sources = [r for r in results if r.score >= max(MIN_SCORE, 0.5 * top)]
        text = self.llm.answer(question, sources)
        return Answer(text, sources, self.llm.name, int((time.perf_counter() - start) * 1000))


def build_service(settings: Settings) -> RagService:
    embedder: Embedder
    if settings.embedding_provider == "openai":
        embedder = OpenAIEmbedder(settings.openai_api_key, settings.openai_embedding_model, settings.embedding_dim)
    else:
        embedder = HashingEmbedder(settings.embedding_dim)

    store: VectorStore
    if settings.vector_store == "pgvector":
        store = PgVectorStore(settings.database_url, settings.embedding_dim)
    else:
        store = InMemoryStore()

    llm: LLM
    if settings.llm_provider == "openai":
        llm = OpenAILLM(settings.openai_api_key, settings.openai_model)
    elif settings.llm_provider == "anthropic":
        llm = AnthropicLLM(settings.anthropic_api_key, settings.anthropic_model)
    else:
        llm = ExtractiveLLM()

    return RagService(embedder, store, llm, settings)

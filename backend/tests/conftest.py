import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.embeddings import HashingEmbedder
from app.llm import ExtractiveLLM
from app.main import create_app, get_service
from app.rag import RagService
from app.store import InMemoryStore


@pytest.fixture
def service() -> RagService:
    settings = Settings(
        llm_provider="extractive", embedding_provider="hashing", vector_store="memory", chunk_size=300, chunk_overlap=50
    )
    return RagService(HashingEmbedder(384), InMemoryStore(), ExtractiveLLM(), settings)


@pytest.fixture
def client(service: RagService) -> TestClient:
    app = create_app()
    app.dependency_overrides[get_service] = lambda: service
    return TestClient(app)

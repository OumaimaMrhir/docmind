"""API FastAPI de DocMind."""

from __future__ import annotations

import logging
from functools import lru_cache

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from . import __version__
from .config import Settings, get_settings
from .loaders import extract_text
from .rag import RagService, build_service

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
logger = logging.getLogger("docmind")

MAX_UPLOAD_BYTES = 5 * 1024 * 1024


@lru_cache
def get_service() -> RagService:
    return build_service(get_settings())


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    expected = get_settings().api_key
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="Clé API invalide ou manquante.")


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    top_k: int | None = Field(default=None, ge=1, le=10)


class SourceOut(BaseModel):
    doc_id: str
    doc_name: str
    chunk_index: int
    content: str
    score: float


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceOut]
    provider: str
    latency_ms: int


class TextDocument(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    content: str = Field(min_length=1)


class IngestResponse(BaseModel):
    doc_id: str
    doc_name: str
    chunks: int


class DocumentOut(BaseModel):
    doc_id: str
    doc_name: str
    chunks: int


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    app = FastAPI(title="DocMind API", version=__version__, description="Assistant RAG sur une base de connaissances interne.")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in settings.cors_origins.split(",") if o.strip()],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "version": __version__, "llm": settings.llm_provider, "store": settings.vector_store}

    @app.get("/documents", response_model=list[DocumentOut], dependencies=[Depends(require_api_key)])
    def list_documents(service: RagService = Depends(get_service)):
        return [DocumentOut(doc_id=d.doc_id, doc_name=d.doc_name, chunks=d.chunks) for d in service.store.list_documents()]

    @app.post("/documents/text", response_model=IngestResponse, status_code=201, dependencies=[Depends(require_api_key)])
    def ingest_text(doc: TextDocument, service: RagService = Depends(get_service)):
        try:
            doc_id, n = service.ingest(doc.name, doc.content)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        logger.info("Document indexé : %s (%d chunks)", doc.name, n)
        return IngestResponse(doc_id=doc_id, doc_name=doc.name, chunks=n)

    @app.post("/documents/upload", response_model=IngestResponse, status_code=201, dependencies=[Depends(require_api_key)])
    async def upload(file: UploadFile = File(...), service: RagService = Depends(get_service)):
        data = await file.read()
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Fichier trop volumineux (5 Mo maximum).")
        try:
            text = extract_text(file.filename or "document", data)
            doc_id, n = service.ingest(file.filename or "document", text)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        logger.info("Fichier indexé : %s (%d chunks)", file.filename, n)
        return IngestResponse(doc_id=doc_id, doc_name=file.filename or "document", chunks=n)

    @app.delete("/documents/{doc_id}", status_code=204, dependencies=[Depends(require_api_key)])
    def delete_document(doc_id: str, service: RagService = Depends(get_service)):
        if not service.store.delete_document(doc_id):
            raise HTTPException(status_code=404, detail="Document introuvable.")

    @app.post("/ask", response_model=AskResponse, dependencies=[Depends(require_api_key)])
    def ask(req: AskRequest, service: RagService = Depends(get_service)):
        result = service.ask(req.question, req.top_k)
        logger.info("Question traitée en %d ms (%s, %d sources)", result.latency_ms, result.provider, len(result.sources))
        return AskResponse(
            answer=result.answer,
            sources=[
                SourceOut(doc_id=s.doc_id, doc_name=s.doc_name, chunk_index=s.index, content=s.content, score=round(s.score, 4))
                for s in result.sources
            ],
            provider=result.provider,
            latency_ms=result.latency_ms,
        )

    return app


app = create_app()

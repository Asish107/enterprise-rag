"""FastAPI application exposing the RAG workflow over HTTP."""
from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile

from app.config import Settings, get_settings
from app.eval import evaluate
from app.ingestion import SUPPORTED_FORMATS
from app.models import EvalRequest, EvalReport, IngestResponse, QueryRequest, QueryResponse
from app.service import RAGService

app = FastAPI(
    title="EnterpriseRAG",
    version="0.1.0",
    description=(
        "Ingestion + retrieval + grounded generation service for PDF, DOCX, and "
        "TXT/MD documents, with citations and production evaluation signals."
    ),
)

_service: RAGService | None = None


def get_service(settings: Settings = Depends(get_settings)) -> RAGService:
    global _service
    if _service is None:
        _service = RAGService(settings)
    return _service


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "enterprise-rag"}


@app.post("/ingest", response_model=IngestResponse)
async def ingest(
    file: UploadFile = File(...),
    service: RAGService = Depends(get_service),
) -> IngestResponse:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported format '{suffix}'. Supported: {sorted(SUPPORTED_FORMATS)}",
        )
    contents = await file.read()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
        tmp.write(contents)
        tmp.flush()
        try:
            return service.ingest_file(tmp.name, filename=file.filename)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest, service: RAGService = Depends(get_service)) -> QueryResponse:
    return service.query(req.question, top_k=req.top_k)


@app.post("/evaluate", response_model=EvalReport)
def run_eval(req: EvalRequest, service: RAGService = Depends(get_service)) -> EvalReport:
    return evaluate(req.samples, lambda q: service.query(q, top_k=req.top_k))

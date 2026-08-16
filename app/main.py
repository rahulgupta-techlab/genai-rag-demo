"""FastAPI app exposing the RAG pipeline as a single POST /query endpoint."""
from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from app.rag_pipeline import RagPipeline

app = FastAPI(title="GenAI RAG Demo")
pipeline: RagPipeline | None = None


class QueryRequest(BaseModel):
    question: str


@app.on_event("startup")
def load_pipeline() -> None:
    global pipeline
    pipeline = RagPipeline()


@app.post("/query")
def query(request: QueryRequest) -> dict:
    assert pipeline is not None, "Pipeline not loaded - run app/ingest.py first"
    return pipeline.answer(request.question)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}

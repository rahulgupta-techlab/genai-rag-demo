"""Retrieval + prompt assembly for the RAG demo.

The LLM call is behind a small `LLMClient` protocol so a real provider
(OpenAI, Azure OpenAI, a local model, etc.) can be dropped in without
touching the retrieval logic.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

import faiss
from sentence_transformers import SentenceTransformer

from app.ingest import EMBEDDING_MODEL, INDEX_DIR


class LLMClient(Protocol):
    def generate(self, question: str, context: str) -> str:
        ...


class EchoLLMClient:
    """Default no-dependency client: returns the most relevant context instead
    of calling a real model, so the pipeline is runnable without API keys."""

    def generate(self, question: str, context: str) -> str:
        return (
            "No LLM configured - showing the retrieved context that would be "
            f"sent to the model for the question: {question!r}\n\n{context}"
        )


class RagPipeline:
    def __init__(self, llm_client: LLMClient | None = None, top_k: int = 3):
        self.top_k = top_k
        self.llm_client = llm_client or EchoLLMClient()
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.index = faiss.read_index(str(INDEX_DIR / "docs.index"))
        with open(INDEX_DIR / "metadata.json", encoding="utf-8") as f:
            self.metadata = json.load(f)

    def retrieve(self, question: str) -> list[dict]:
        embedding = self.model.encode([question], convert_to_numpy=True)
        faiss.normalize_L2(embedding)
        scores, indices = self.index.search(embedding, self.top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            doc = dict(self.metadata[idx])
            doc["score"] = float(score)
            results.append(doc)
        return results

    def answer(self, question: str) -> dict:
        retrieved = self.retrieve(question)
        context = "\n\n".join(f"[{doc['source']}#{doc['chunk_id']}] {doc['text']}" for doc in retrieved)
        response = self.llm_client.generate(question, context)
        return {"question": question, "answer": response, "sources": retrieved}

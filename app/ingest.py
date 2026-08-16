"""Chunk the sample documents, embed them, and build a local FAISS index."""
from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "sample_docs"
INDEX_DIR = Path(__file__).resolve().parent.parent / "data" / "index"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    words = text.split()
    if not words:
        return []
    chunks = []
    step = max(chunk_size - overlap, 1)
    for start in range(0, len(words), step):
        chunk = " ".join(words[start : start + chunk_size])
        if chunk:
            chunks.append(chunk)
        if start + chunk_size >= len(words):
            break
    return chunks


def load_documents() -> list[dict]:
    documents = []
    for path in sorted(DATA_DIR.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        for i, chunk in enumerate(chunk_text(text)):
            documents.append({"source": path.name, "chunk_id": i, "text": chunk})
    return documents


def build_index(documents: list[dict], model: SentenceTransformer) -> faiss.Index:
    embeddings = model.encode([doc["text"] for doc in documents], convert_to_numpy=True)
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


def main() -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    documents = load_documents()
    if not documents:
        raise SystemExit(f"No documents found in {DATA_DIR}")

    model = SentenceTransformer(EMBEDDING_MODEL)
    index = build_index(documents, model)

    faiss.write_index(index, str(INDEX_DIR / "docs.index"))
    with open(INDEX_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)

    print(f"Indexed {len(documents)} chunks from {DATA_DIR} into {INDEX_DIR}")


if __name__ == "__main__":
    main()

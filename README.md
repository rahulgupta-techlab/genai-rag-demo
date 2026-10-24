# GenAI RAG Demo

A small, self-contained Retrieval-Augmented Generation (RAG) pipeline: ingest text
documents, chunk and embed them, store the embeddings in a local FAISS index, and
answer questions by retrieving the most relevant chunks and passing them to an LLM
as context.

Built to demonstrate the core RAG pattern end-to-end without external services -
everything runs locally with open-source embedding models. The generation step is
pluggable: swap in OpenAI, Azure OpenAI, or a local model by implementing
`LLMClient` in `app/rag_pipeline.py`.

## Stack

- Python 3.11
- `sentence-transformers` for embeddings
- `faiss-cpu` for vector search
- FastAPI for the query endpoint

## Project layout

```
app/
  ingest.py        # chunk + embed documents, build the FAISS index
  rag_pipeline.py   # retrieval + prompt assembly + LLM call
  main.py           # FastAPI app exposing POST /query
data/sample_docs/   # example source documents
tests/               # unit tests for chunking and retrieval
```

## Running it

```bash
pip install -r requirements.txt
python app/ingest.py                 # builds the local FAISS index from data/sample_docs
uvicorn app.main:app --reload        # serves POST /query
```

```bash
curl -X POST http://localhost:8000/query -H "Content-Type: application/json" \
  -d '{"question": "What does the pipeline do with new documents?"}'
```

## Tests

```bash
pytest tests/
```

## Notes

This is a portfolio/demo project - the sample documents and LLM client are
intentionally minimal so the RAG mechanics (chunking, embedding, retrieval,
context assembly) are easy to follow and extend.

## License

Copyright (c) 2026 Rahul Gupta. All rights reserved. Published for viewing and evaluation only; copying or reuse requires written permission. See [LICENSE](LICENSE).

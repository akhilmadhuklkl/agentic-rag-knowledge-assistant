# Agentic RAG Knowledge Assistant

A local-first AI knowledge assistant that searches trusted documents, verifies evidence, and answers with citations.

## What it does

```text
Question -> Agent plan -> Hybrid retrieval -> Grounded generation -> Verification -> Cited answer
```

The project is a product foundation. Core capabilities live in Python services and FastAPI. Streamlit is retained as a thin backup interface.

## Current capabilities

- PDF, Markdown, and text ingestion
- Chunking with source metadata
- Deterministic local embeddings and persistent vector storage
- Hybrid semantic plus keyword retrieval
- Source and file-type filters
- Bounded planning and re-querying
- Deterministic or local Ollama answer generation
- Citation verification and safe refusal
- SQLite conversations, document metadata, and index fingerprints
- Upload, delete, rebuild, and stale-index cleanup workflows
- Optional API-key and workspace isolation
- Structured logs, request IDs, metrics, retries, and evaluation tests

## Quick start

```powershell
python -m pip install -e ".[dev]"
uvicorn agentic_rag.api.app:app --host 127.0.0.1 --port 8765
```

In another terminal:

```powershell
$env:ASSISTANT_API_URL = "http://127.0.0.1:8765"
python scripts/smoke_test.py
```

API documentation is available at `http://127.0.0.1:8765/docs`.

## Optional Streamlit backup UI

```powershell
python -m pip install -e ".[ui]"
$env:ASSISTANT_API_URL = "http://127.0.0.1:8765"
streamlit run streamlit_backup/app.py
```

The UI calls the API and does not own core RAG logic.

## Configuration

Copy `.env.example` to `.env` when custom settings are needed:

```env
LLM_PROVIDER=deterministic
EMBEDDING_PROVIDER=local
EMBEDDING_MODEL=nomic-embed-text
LLM_MODEL=llama3.2
LLM_BASE_URL=http://localhost:11434
DATABASE_PATH=data/assistant.db
VECTOR_STORE_PATH=data/vectorstore
DOCUMENT_STORAGE_PATH=data/documents
```

Set `LLM_PROVIDER=ollama` to use a locally running Ollama model.

For model-backed retrieval, install and run Ollama, pull an embedding model such as
`nomic-embed-text`, then set `EMBEDDING_PROVIDER=ollama` and re-index the documents.
The default `local` provider remains available without any external service.

## Verification

```powershell
python -m pytest -q
ruff check .
```

Evaluation cases are in [evaluation/sample_cases.json](evaluation/sample_cases.json).
The multi-document benchmark is in [evaluation/multi_document_cases.json](evaluation/multi_document_cases.json), with isolated fixtures under [evaluation/fixtures](evaluation/fixtures).

## Documentation

- [Project scope](docs/project-scope.md)
- [Architecture](docs/architecture.md)
- [API guide](docs/api.md)
- [Streamlit backup UI](docs/streamlit-backup.md)

## Deployment

Hosting, domains, cloud infrastructure, and production operations are intentionally deferred until the local product behavior is complete and evaluated.

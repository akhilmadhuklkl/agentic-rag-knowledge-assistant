# Product Readiness Report

Audit date: 2026-11-14

## Overall status

**Local MVP ready for continued feature development.** The product is not yet deployment-ready; deployment remains intentionally deferred.

## Verified capabilities

- Document ingestion for PDF, Markdown, and text files
- Chunking and source metadata preservation
- Local deterministic embeddings
- Persistent vector storage
- Hybrid semantic and keyword retrieval
- Source and file-type filtering
- Bounded agent planning and retry search
- Deterministic answer generation
- Optional local Ollama generation
- Citation creation and verification
- Safe refusal for unsupported questions
- Conversation IDs and multi-turn history
- SQLite persistence for conversations and document metadata
- Document upload, delete, rebuild, and stale-index cleanup
- Optional API key and workspace isolation
- Request IDs, structured logs, metrics, and timing
- Streamlit backup interface
- Evaluation dataset and quality metrics

## Verification evidence

```text
pytest: 61 passed
ruff: All checks passed
pip check: No broken requirements found
live API smoke test: passed
git status: clean and synchronized with origin/main
```

The live smoke test exercised health, indexing, and a grounded cited answer through the real HTTP service.

## Current limitations

1. The default embedding provider is deterministic feature hashing, not a trained semantic embedding model.
2. The vector store is local JSON rather than a production vector database.
3. Conversation and document persistence currently use SQLite locally.
4. The Ollama provider requires a separately running local model service.
5. Web search and external structured APIs are not connected yet.
6. Workspace isolation is currently configuration/header based, not full multi-tenant authorization.
7. The Streamlit interface is intentionally a backup UI, not the final product frontend.

## Recommended next priorities

### Priority 1: retrieval quality

Add a model-backed embedding provider, reranking, richer evaluation cases, and regression thresholds.

### Priority 2: source coverage

Add web search and structured API tools behind the existing tool/orchestrator interfaces.

### Priority 3: authorization depth

Replace static API-key/workspace configuration with a real identity provider and per-user document permissions.

### Priority 4: production persistence

Move repository implementations to PostgreSQL and the vector store to a managed or database-backed implementation.

### Priority 5: product UI

Build the primary web UI against the API while preserving Streamlit as an internal fallback.

## Deployment gate

Deployment should begin only after the local product has:

- A trained embedding provider selected and evaluated.
- A larger golden evaluation dataset.
- Explicit permission tests for documents and conversations.
- Production database and vector-store choices.
- Secret management and operational monitoring decisions.
- A deployment-specific smoke and rollback plan.

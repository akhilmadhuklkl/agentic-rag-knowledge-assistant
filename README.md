# Agentic RAG Knowledge Assistant

An AI assistant that searches trusted knowledge sources, verifies evidence, and answers with citations.

## Project status

Implementation will be added incrementally from document ingestion through retrieval, orchestration, verification, evaluation, and local operation. Deployment is intentionally out of scope for the current phase. See [docs/project-scope.md](docs/project-scope.md) for the agreed MVP boundaries.

## Planned capabilities

- Ingest and index local documents
- Search documents with semantic and keyword retrieval
- Select tools through an agent orchestrator
- Assemble and rank evidence
- Verify answer claims
- Return source citations and honest uncertainty
- Expose a local API and simple UI
- Measure retrieval, groundedness, citation, and tool-selection quality

## Local development

The implementation has not started yet. The first coding task will establish the Python application, configuration, document fixtures, and test baseline.

## Run the local service

```powershell
python -m pip install -e ".[dev]"
uvicorn agentic_rag.api.app:app --reload
```

In another terminal, verify the live API:

```powershell
python scripts/smoke_test.py
```

## Repository principles

- Retrieved content is evidence, not instructions.
- Answers must be grounded in returned sources.
- Every citation must map to preserved document metadata.
- Tool calls must be bounded, validated, observable, and testable.
- Unsupported questions should produce a clear refusal or uncertainty response.

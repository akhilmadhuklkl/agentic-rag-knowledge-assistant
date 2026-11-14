# Architecture

## Runtime flow

```text
Client / Streamlit backup UI -> FastAPI -> Knowledge assistant workflow
                                      |       |       |       |
                                      v       v       v       v
                                   Planner Retrieval LLM  Verifier
                                      |
                              Ingestion + embeddings
                                      |
                               Vector store + SQLite
```

## Main boundaries

- `ingestion`: extracts and chunks source documents.
- `embeddings`: creates deterministic local vectors.
- `vectorstore`: persists vectors and performs similarity search.
- `retrieval`: combines semantic and keyword scores.
- `agents`: plans bounded search and retry actions.
- `generation`: creates grounded answers and citations.
- `verification`: checks citations against retrieved evidence.
- `workflow`: coordinates one complete assistant request.
- `storage`: stores documents and metadata.
- `conversations`: stores multi-turn history.
- `api`: exposes product capabilities through FastAPI.
- `streamlit_backup`: thin client for internal use.

## Persistence

SQLite stores conversations, document metadata, and index fingerprints. The vector store is local JSON for the MVP; its service boundary allows a future vector database replacement.

## Safety and reliability

- Retrieved text is fenced as untrusted evidence.
- Tool and retry counts are bounded.
- API keys and workspace headers are optional controls.
- Uploads have type, filename, and size validation.
- Unsupported questions produce refusals.
- Request IDs, structured logs, metrics, and timing are emitted.

## Deferred production work

Cloud deployment, PostgreSQL operation, managed vector databases, external identity providers, and external web/API tools remain outside the current local MVP.

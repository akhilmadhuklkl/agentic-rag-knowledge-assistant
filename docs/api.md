# API Guide

Start the service:

```powershell
uvicorn agentic_rag.api.app:app --host 127.0.0.1 --port 8765
```

## Index and ask

```powershell
Invoke-RestMethod -Method Post http://127.0.0.1:8765/knowledge/index/file `
  -ContentType "application/json" -Body '{"path":"tests/fixtures/policy.md"}'

Invoke-RestMethod -Method Post http://127.0.0.1:8765/knowledge/ask `
  -ContentType "application/json" -Body '{"query":"What is the refund policy?","top_k":3}'
```

The answer response includes `answer`, `citations`, `grounded`, `verified`, `attempts`, and `conversation_id`.

## Search and document lifecycle

```text
POST   /knowledge/search
POST   /knowledge/documents/upload
GET    /knowledge/documents
DELETE /knowledge/documents/{name}
POST   /knowledge/index/rebuild
POST   /knowledge/index/cleanup
```

## Conversation lifecycle

```text
GET    /knowledge/conversations
GET    /knowledge/conversations/{conversation_id}
DELETE /knowledge/conversations/{conversation_id}
```

## System endpoints

```text
GET /health
GET /metrics
```

If `API_KEY` is configured, send `X-API-Key`. If workspace enforcement is enabled, send `X-Workspace-ID` too.

# Development and operations guide

## Product flow

Atlas follows this path for a question:

```text
User question
  -> agent orchestration and query planning
  -> hybrid document retrieval
  -> grounded answer generation
  -> citation and refusal verification
  -> animated, cited answer in the product UI
```

Uploaded PDF, Markdown, and text files are stored under the configured document path and indexed into the configured vector-store path. The SQLite database stores conversations, document metadata, ingestion activity, latency history, feedback, and evaluation history.

## Daily local workflow

1. Start the API:

   ```powershell
   uvicorn agentic_rag.api.app:app --host 127.0.0.1 --port 8765
   ```

2. Open `http://127.0.0.1:8765/app/`.
3. Upload or refresh documents in the Knowledge Base panel.
4. Ask questions and inspect citations, confidence, retrieval diagnostics, and the agent trace.
5. Run the sample or multi-document quality benchmark.
6. Export the latest report as JSON or CSV when sharing results.

## Quality evaluation

The evaluation panel measures retrieval, citations, answer quality, refusal behavior, category scores, historical trends, and threshold status. Thresholds can be adjusted in the Quality Gates controls for the next run.

Benchmark data lives in `evaluation/`. Add a case with a precise `expected_source` or an `expected_sources` list for multi-hop questions, answer keywords, `expected_refusal`, and an optional `category`. Keep refusal cases clearly outside the indexed knowledge so they test grounding rather than vocabulary overlap. Multi-hop questions joined by “and” are decomposed into clause searches and their evidence is merged before answer generation.

## Testing before a commit

```powershell
python -m pytest -q
ruff check src tests
node --check frontend/app.js
git diff --check
```

The frontend smoke tests in `tests/test_frontend_smoke.py` verify that the critical static UI hooks remain present. GitHub Actions repeats these checks automatically.

## Ollama operation

When using Ollama, confirm the service is reachable and the configured models are available:

```powershell
ollama list
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

Set `LLM_PROVIDER=ollama` and `EMBEDDING_PROVIDER=ollama` in `.env`, then re-index documents after changing the embedding provider or model.

For local request boundaries, set `API_KEY` and `WORKSPACE_ID`. When `REQUIRE_WORKSPACE_HEADER=true`, every protected request must include `X-Workspace-ID`; callers may also provide `X-User-ID` for request-level identity logging. Workspace and user identifiers accept letters, numbers, `_`, and `-` only. Full identity-provider authentication and per-document ownership are deployment-phase work.

Retrieval tuning is controlled by `RETRIEVAL_SEMANTIC_WEIGHT`, `RETRIEVAL_RERANK_WEIGHT`, `MINIMUM_EVIDENCE_SCORE`, and `MINIMUM_LEXICAL_OVERLAP`. Increase rerank weight to favor query terms occurring close together; increase the minimum thresholds to require stronger evidence, with the trade-off that more questions may safely refuse.

Web fallback is opt-in through `WEB_SEARCH_ENABLED=true`. Restrict it with `WEB_SEARCH_ALLOWED_DOMAINS=example.com,docs.example.com`; an empty allowlist permits results from the configured search provider.

## Backup and local data

`streamlit_backup/` is intentionally preserved as a fallback UI. Do not remove it while the product frontend is evolving. Local runtime data is stored under `data/` and is ignored by Git; preserve it separately if conversations, uploaded files, or evaluation history need to survive a machine reset.

## Troubleshooting

- No sources: check the Knowledge Base health panel and re-index stale documents.
- Provider attention needed: inspect `.env`, confirm Ollama is running, and check `ollama list`.
- Low confidence: open the cited source excerpt and adjust retrieval mode, source count, or evidence strictness.
- Evaluation needs review: inspect the failed quality gates and case details before changing thresholds.
- UI not updated: restart Uvicorn so the latest static frontend files are served.

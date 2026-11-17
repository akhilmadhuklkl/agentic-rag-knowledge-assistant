# Recycle-bin operations

The knowledge assistant uses a recycle bin so document deletion is recoverable by default.

## Configuration

Set these environment variables in `.env` when needed:

- `RECYCLE_BIN_RETENTION_DAYS`: age after which deleted files are eligible for permanent cleanup. Default: `30`.
- `RECYCLE_BIN_CLEANUP_INTERVAL_SECONDS`: automatic cleanup interval. Default: `86400` (24 hours).
- `RECYCLE_BIN_CLEANUP_ALERT_THRESHOLD_MS`: duration threshold for a slow-cleanup alert. Default: `5000`.

Values are validated at startup. Retention and interval values must be between 1 minute/1 day limits and 3650 days, as defined by the application settings.

## User workflow

1. Deleting a document moves it to `data/recycle_bin` and removes it from the active document repository and vector index.
2. The document appears under the UI's Recovery section.
3. `Restore & re-index` moves it back to active storage and rebuilds its indexed knowledge.
4. Restoring is blocked when an active document with the same filename already exists.

## Administrator cleanup

Administrators can:

- Run cleanup immediately using the configured retention period.
- Permanently delete one recycle-bin item.
- Permanently purge items older than a selected age.
- View cleanup status, history, summaries, charts, performance, and alerts.
- Export filtered cleanup history as JSON or CSV.

Permanent deletion cannot be undone, so the UI requires confirmation.

## Monitoring endpoints

- `GET /health/recycle-bin`: current scheduler state, retention, interval, latest result, and alert.
- `POST /health/recycle-bin/run`: manually run cleanup; administrator authorization required.
- `GET /health/recycle-bin/history?limit=10&offset=0`: paginated cleanup history.
- `GET /health/recycle-bin/summary`: aggregate run, removal, success, failure, and duration statistics.
- `GET /health/recycle-bin/export?format=json|csv`: download cleanup history, optionally filtered by `from_date`, `to_date`, and `status`.

## Verification checklist

Run these checks before handing off a change:

```powershell
ruff check .
python -m pytest -q
git diff --check
git status --short --branch
```

The cleanup worker is tested with fake storage so tests do not wait for the production interval. The worker uses FastAPI lifespan handling and is cancelled cleanly during shutdown.

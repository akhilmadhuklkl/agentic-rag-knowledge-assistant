"""Knowledge-base API routes."""

import hashlib
import json
import math
import re
from pathlib import Path
from tempfile import TemporaryDirectory

import httpx
from fastapi import APIRouter, Depends, File, Query, Request, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse

from agentic_rag.agents.orchestrator import AgentOrchestrator
from agentic_rag.api.schemas import (
    AdminSettingsResponse,
    AnswerRequest,
    AnswerResponse,
    AskResponse,
    BulkDocumentMetadataResponse,
    BulkDocumentMetadataUndo,
    BulkDocumentMetadataUpdate,
    CitationResponse,
    CleanupResponse,
    ConversationListResponse,
    ConversationResponse,
    ConversationTurnResponse,
    DocumentListResponse,
    DocumentMetadataUpdate,
    DocumentResponse,
    EvaluationHistoryResponse,
    EvaluationReportResponse,
    EvaluationRunRequest,
    FeedbackRequest,
    IndexPathRequest,
    IndexResponse,
    IngestionActivityResponse,
    KnowledgeBaseHealthResponse,
    LatencyHistoryResponse,
    LatencySummaryResponse,
    ProviderComparisonResponse,
    ProviderHealthResponse,
    ReindexStaleResponse,
    SearchRequest,
    SearchResponse,
    SearchResultResponse,
    SecurityAuditEventResponse,
)
from agentic_rag.config import Settings, get_settings
from agentic_rag.conversations.repository import SQLiteConversationRepository
from agentic_rag.conversations.store import ConversationStore
from agentic_rag.embeddings.local import LocalHashEmbeddingProvider
from agentic_rag.embeddings.providers import build_embedding_provider
from agentic_rag.evaluation.service import EvaluationService
from agentic_rag.generation.providers import OllamaAnswerModel, build_answer_model
from agentic_rag.generation.service import GroundedAnswerService
from agentic_rag.indexing.service import DocumentIndexingService
from agentic_rag.metrics import metrics
from agentic_rag.observability import log_event
from agentic_rag.storage.documents import DocumentStorageService, StoredDocument
from agentic_rag.storage.repository import (
    SQLiteAuditRepository,
    SQLiteDocumentRepository,
    SQLiteEvaluationRepository,
    SQLiteFeedbackRepository,
    SQLiteIngestionRepository,
    SQLiteLatencyRepository,
)
from agentic_rag.tools.web_search import WebSearchTool
from agentic_rag.workflow.service import KnowledgeAssistantService

router = APIRouter(prefix="/knowledge", tags=["knowledge"])
conversation_store = ConversationStore(
    repository=SQLiteConversationRepository(get_settings().database_path)
)
document_repository = SQLiteDocumentRepository(get_settings().database_path)
feedback_repository = SQLiteFeedbackRepository(get_settings().database_path)
evaluation_repository = SQLiteEvaluationRepository(get_settings().database_path)
ingestion_repository = SQLiteIngestionRepository(get_settings().database_path)
latency_repository = SQLiteLatencyRepository(get_settings().database_path)
audit_repository = SQLiteAuditRepository(get_settings().database_path)


def _search_mode_weight(mode: str, hybrid_weight: float) -> float:
    """Translate a user-facing retrieval mode into a ranking weight."""

    return {"semantic": 1.0, "keyword": 0.0}.get(mode, hybrid_weight)


def _latency_target(profile: str, default: float) -> float:
    """Translate a performance profile into a warning target."""

    return {"local": default * 1.5, "strict": default * 0.6}.get(profile, default)


@router.get("/providers/health", response_model=ProviderHealthResponse)
def provider_health(settings: Settings = Depends(get_settings)) -> ProviderHealthResponse:  # noqa: B008
    """Report configured model providers and Ollama reachability."""

    needs_ollama = settings.llm_provider.lower() == "ollama" or settings.embedding_provider.lower() == "ollama"
    reachable = False
    if needs_ollama:
        try:
            response = httpx.get(f"{settings.llm_base_url.rstrip('/')}/api/tags", timeout=2.0)
            reachable = response.is_success
        except httpx.HTTPError:
            reachable = False
    else:
        reachable = True
    embedding_provider = build_embedding_provider(settings)
    embedding_provider_id = getattr(embedding_provider, "provider_id", type(embedding_provider).__name__)
    metadata_path = Path(settings.vector_store_path).with_suffix(".meta.json")
    indexed_provider_id = None
    index_status = "not_created"
    if metadata_path.is_file():
        try:
            indexed_provider_id = json.loads(metadata_path.read_text(encoding="utf-8")).get("provider_id")
        except (OSError, json.JSONDecodeError):
            index_status = "needs_reindex"
        else:
            index_status = "compatible" if indexed_provider_id == embedding_provider_id else "needs_reindex"
    status = "ready" if reachable and index_status != "needs_reindex" else "degraded"
    if index_status == "needs_reindex":
        detail = "Embedding provider changed; re-index the knowledge base before asking questions."
    elif reachable and not needs_ollama:
        detail = "Configured local providers are ready."
    elif reachable:
        detail = "Ollama is reachable and the configured embedding index is compatible."
    else:
        detail = "Ollama is not reachable; check that the local service is running."
    return ProviderHealthResponse(status=status, llm_provider=settings.llm_provider, llm_model=settings.llm_model, embedding_provider=settings.embedding_provider, embedding_model=settings.embedding_model, ollama_reachable=reachable, detail=detail, embedding_provider_id=embedding_provider_id, indexed_provider_id=indexed_provider_id, index_status=index_status)


@router.get("/security/audit", response_model=list[SecurityAuditEventResponse])
def security_audit(request: Request, settings: Settings = Depends(get_settings), limit: int = Query(default=20, ge=1, le=100), offset: int = Query(default=0, ge=0), query: str | None = Query(default=None, max_length=100)) -> list[SecurityAuditEventResponse]:  # noqa: B008
    """Return recent authentication audit events without sensitive headers."""

    admins = {value.strip() for value in settings.admin_user_ids.split(",") if value.strip()}
    _require_admin(request, admins)
    return [SecurityAuditEventResponse(**event) for event in audit_repository.recent(limit, offset, query)]


@router.get("/admin/settings", response_model=AdminSettingsResponse)
def admin_settings(request: Request, settings: Settings = Depends(get_settings)) -> AdminSettingsResponse:  # noqa: B008
    """Expose safe administrative configuration to authorized administrators."""

    admins = {value.strip() for value in settings.admin_user_ids.split(",") if value.strip()}
    _require_admin(request, admins)
    return AdminSettingsResponse(
        audit_retention_days=settings.audit_retention_days,
        configured_admin_count=len(admins),
        admin_configuration="environment-managed",
    )


def _require_admin(request: Request, admins: set[str]) -> None:
    user_id = getattr(getattr(request, "state", None), "user_id", "anonymous")
    if admins and user_id not in admins:
        raise PermissionError("Administrator access required")


def get_indexing_service(settings: Settings = Depends(get_settings)) -> DocumentIndexingService:  # noqa: B008
    """Build the application service from runtime configuration."""

    return DocumentIndexingService(
        settings.vector_store_path,
        embedding_provider=build_embedding_provider(settings),
        semantic_weight=settings.retrieval_semantic_weight,
        rerank_weight=settings.retrieval_rerank_weight,
        metadata_repository=document_repository,
    )


def get_document_storage(settings: Settings = Depends(get_settings)) -> DocumentStorageService:  # noqa: B008
    """Build the controlled local document storage service."""

    return DocumentStorageService(
        settings.document_storage_path,
        max_size_bytes=settings.max_upload_size_bytes,
        repository=document_repository,
    )


@router.get("/documents", response_model=DocumentListResponse)
def list_documents(
    storage: DocumentStorageService = Depends(get_document_storage),  # noqa: B008
    service: DocumentIndexingService = Depends(get_indexing_service),  # noqa: B008
) -> DocumentListResponse:
    """List supported documents currently stored locally."""

    return DocumentListResponse(documents=[_document_response(document, service) for document in storage.list_documents()])


def _document_response(document: object, service: DocumentIndexingService) -> DocumentResponse:
    """Attach current index readiness to one stored document."""

    path = Path(document.path)
    if not path.is_file():
        status = "missing"
    elif document.path not in service.vector_store.source_paths():
        status = "stale"
    else:
        fingerprint = hashlib.sha256(path.read_bytes()).hexdigest()
        status = "indexed" if fingerprint == document_repository.get_fingerprint(document.path) else "stale"
    return DocumentResponse(**document.__dict__, status=status, upload_date=document_repository.get_created_at(document.name))


@router.get("/documents/health", response_model=KnowledgeBaseHealthResponse)
def knowledge_base_health(
    service: DocumentIndexingService = Depends(get_indexing_service),  # noqa: B008
) -> KnowledgeBaseHealthResponse:
    """Report whether stored documents are represented in the current index."""

    documents = document_repository.list_documents()
    indexed_paths = service.vector_store.source_paths()
    stale = 0
    for document in documents:
        path = Path(document.path)
        fingerprint = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        if fingerprint != document_repository.get_fingerprint(document.path) or document.path not in indexed_paths:
            stale += 1
    status = "empty" if not documents else "needs_reindex" if stale else "ready"
    return KnowledgeBaseHealthResponse(
        document_count=len(documents),
        indexed_document_count=len(indexed_paths),
        chunk_count=service.vector_store.count(),
        stale_document_count=stale,
        status=status,
    )


@router.post("/documents/reindex-stale", response_model=ReindexStaleResponse)
def reindex_stale_documents(
    service: DocumentIndexingService = Depends(get_indexing_service),  # noqa: B008
) -> ReindexStaleResponse:
    """Re-index changed sources while leaving missing files untouched."""

    reindexed = 0
    missing = 0
    processed: list[str] = []
    errors: list[str] = []
    for document in document_repository.list_documents():
        path = Path(document.path)
        if not path.is_file():
            missing += 1
            continue
        fingerprint = hashlib.sha256(path.read_bytes()).hexdigest()
        if fingerprint != document_repository.get_fingerprint(document.path) or document.path not in service.vector_store.source_paths():
            try:
                service.reindex_file(document.path)
                processed.append(document.name)
                ingestion_repository.record("reindexed", document.name, "Bulk stale-document refresh completed")
                reindexed += 1
            except (OSError, ValueError, RuntimeError) as error:
                errors.append(f"{document.name}: {error}")
                ingestion_repository.record("failed", document.name, str(error))
    return ReindexStaleResponse(
        reindexed_count=reindexed,
        skipped_missing_count=missing,
        chunk_count=service.vector_store.count(),
        processed_documents=processed,
        errors=errors,
    )


@router.post("/documents/upload", response_model=DocumentResponse)
async def upload_document(
    file: UploadFile = File(...),  # noqa: B008
    storage: DocumentStorageService = Depends(get_document_storage),  # noqa: B008
    indexing: DocumentIndexingService = Depends(get_indexing_service),  # noqa: B008
) -> DocumentResponse:
    """Upload one supported document into local controlled storage."""

    document = storage.save(file.filename or "", await file.read())
    summary = indexing.index_file(document.path)
    ingestion_repository.record("uploaded", document.name, f"Indexed {summary.chunk_count} chunks")
    return DocumentResponse(**document.__dict__)


@router.delete("/documents/{name}", response_model=DocumentResponse)
def delete_document(
    name: str,
    storage: DocumentStorageService = Depends(get_document_storage),  # noqa: B008
    indexing: DocumentIndexingService = Depends(get_indexing_service),  # noqa: B008
) -> DocumentResponse:
    """Delete a document and its indexed chunks."""

    document = storage.delete(name)
    indexing.vector_store.delete_source(document.path)
    ingestion_repository.record("deleted", document.name, "Removed document and indexed knowledge")
    return DocumentResponse(**document.__dict__)


@router.patch("/documents/{name}/metadata", response_model=DocumentResponse)
def update_document_metadata(
    name: str,
    request: DocumentMetadataUpdate,
    storage: DocumentStorageService = Depends(get_document_storage),  # noqa: B008
    indexing: DocumentIndexingService = Depends(get_indexing_service),  # noqa: B008
) -> DocumentResponse:
    """Update labels used to prioritize and filter a document."""

    document = next((item for item in document_repository.list_documents() if item.name == name), None)
    if document is None:
        raise FileNotFoundError(name)
    document_repository.update_metadata(name, request.authority, request.freshness)
    indexing.reindex_file(document.path)
    indexing.vector_store.update_source_metadata(document.path, request.authority, request.freshness)
    updated = StoredDocument(document.name, document.path, document.size_bytes, document.file_type, request.authority, request.freshness)
    ingestion_repository.record("metadata_updated", name, f"Authority={document.authority}->{request.authority}; freshness={document.freshness}->{request.freshness}")
    return DocumentResponse(**updated.__dict__)


@router.patch("/documents/metadata", response_model=BulkDocumentMetadataResponse)
def update_documents_metadata(
    request: BulkDocumentMetadataUpdate,
    indexing: DocumentIndexingService = Depends(get_indexing_service),  # noqa: B008
) -> list[DocumentResponse]:
    """Apply labels and refresh indexes for several documents."""

    available = {item.name: item for item in document_repository.list_documents()}
    updated_documents: list[DocumentResponse] = []
    errors: list[str] = []
    for name in request.names:
        document = available.get(name)
        if document is None:
            errors.append(f"{name}: document not found")
            continue
        document_repository.update_metadata(name, request.authority, request.freshness)
        indexing.reindex_file(document.path)
        indexing.vector_store.update_source_metadata(document.path, request.authority, request.freshness)
        ingestion_repository.record("metadata_updated", name, f"Authority={document.authority}->{request.authority}; freshness={document.freshness}->{request.freshness}")
        updated_documents.append(DocumentResponse(**StoredDocument(document.name, document.path, document.size_bytes, document.file_type, request.authority, request.freshness).__dict__))
    return BulkDocumentMetadataResponse(updated=updated_documents, errors=errors)


@router.post("/documents/{name}/metadata-history/undo", response_model=DocumentResponse)
def undo_document_metadata(
    name: str,
    indexing: DocumentIndexingService = Depends(get_indexing_service),  # noqa: B008
) -> DocumentResponse:
    """Undo the latest metadata edit when its previous labels are available."""

    document = next((item for item in document_repository.list_documents() if item.name == name), None)
    if document is None:
        raise FileNotFoundError(name)
    history = ingestion_repository.recent_for_document(name, limit=1)
    if not history:
        raise ValueError("No metadata edit is available to undo")
    detail = history[0]["detail"]
    authority_match = re.search(r"Authority=([a-z]+)->([a-z]+)", detail)
    freshness_match = re.search(r"freshness=([a-z]+)->([a-z]+)", detail)
    if not authority_match or not freshness_match:
        raise ValueError("The latest metadata edit cannot be undone")
    authority, freshness = authority_match.group(1), freshness_match.group(1)
    document_repository.update_metadata(name, authority, freshness)
    indexing.reindex_file(document.path)
    indexing.vector_store.update_source_metadata(document.path, authority, freshness)
    ingestion_repository.record("metadata_undone", name, f"Restored authority={authority}; freshness={freshness}")
    return DocumentResponse(**StoredDocument(document.name, document.path, document.size_bytes, document.file_type, authority, freshness).__dict__)


@router.post("/documents/metadata-history/undo", response_model=BulkDocumentMetadataResponse)
def undo_documents_metadata(
    request: BulkDocumentMetadataUndo,
    indexing: DocumentIndexingService = Depends(get_indexing_service),  # noqa: B008
) -> BulkDocumentMetadataResponse:
    """Undo the latest metadata edit for each selected document."""

    updated: list[DocumentResponse] = []
    errors: list[str] = []
    for name in request.names:
        try:
            updated.append(undo_document_metadata(name, indexing))
        except (FileNotFoundError, ValueError) as error:
            errors.append(f"{name}: {error}")
    return BulkDocumentMetadataResponse(updated=updated, errors=errors)


def _stored_document_path(name: str, storage: DocumentStorageService) -> Path:
    """Resolve a stored filename without allowing path traversal."""

    safe_name = storage._safe_filename(name)
    path = (storage.root / safe_name).resolve()
    if not path.is_relative_to(storage.root.resolve()) or not path.is_file():
        raise FileNotFoundError(name)
    return path


@router.get("/documents/{name}/download")
def download_document(
    name: str,
    storage: DocumentStorageService = Depends(get_document_storage),  # noqa: B008
) -> FileResponse:
    """Download an original stored document."""

    path = _stored_document_path(name, storage)
    return FileResponse(path, filename=path.name, media_type="application/octet-stream")


@router.get("/documents/{name}/preview")
def preview_document(
    name: str,
    page: int = Query(default=1, ge=1, description="1-based PDF page number"),
    storage: DocumentStorageService = Depends(get_document_storage),  # noqa: B008
) -> PlainTextResponse:
    """Return a safe text preview, including a selected PDF page."""

    path = _stored_document_path(name, storage)
    if path.suffix.lower() in {".md", ".markdown", ".txt"}:
        return PlainTextResponse(path.read_text(encoding="utf-8"))
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        if page > len(reader.pages):
            raise ValueError(f"PDF has {len(reader.pages)} page(s); page {page} does not exist")
        text = reader.pages[page - 1].extract_text() or ""
        return PlainTextResponse(text, headers={"X-Preview-Page": str(page), "X-Preview-Total-Pages": str(len(reader.pages))})
    raise ValueError("Preview is available for supported document types only.")


@router.post("/index/file", response_model=IndexResponse)
def index_file(
    request: IndexPathRequest,
    service: DocumentIndexingService = Depends(get_indexing_service),  # noqa: B008
) -> IndexResponse:
    """Index one local document."""

    summary = service.index_file(request.path)
    ingestion_repository.record("reindexed", Path(request.path).name, f"Indexed {summary.chunk_count} chunks")
    return IndexResponse(**summary.__dict__)


@router.get("/documents/activity", response_model=list[IngestionActivityResponse])
def ingestion_activity() -> list[IngestionActivityResponse]:
    """Return recent document ingestion operations."""

    return [IngestionActivityResponse(**event) for event in ingestion_repository.recent()]


@router.get("/documents/{name}/metadata-history", response_model=list[IngestionActivityResponse])
def document_metadata_history(name: str) -> list[IngestionActivityResponse]:
    """Return the metadata edit audit trail for one document."""

    return [IngestionActivityResponse(**event) for event in ingestion_repository.recent_for_document(name)]


@router.post("/index/directory", response_model=IndexResponse)
def index_directory(
    request: IndexPathRequest,
    service: DocumentIndexingService = Depends(get_indexing_service),  # noqa: B008
) -> IndexResponse:
    """Index supported documents in one local directory."""

    summary = service.index_directory(request.path)
    return IndexResponse(**summary.__dict__)


@router.post("/index/rebuild", response_model=IndexResponse)
def rebuild_index(
    request: IndexPathRequest,
    service: DocumentIndexingService = Depends(get_indexing_service),  # noqa: B008
) -> IndexResponse:
    """Force re-index all supported files in a directory."""

    root = Path(request.path)
    total = service.index_directory(root)
    for path in sorted(root.iterdir()):
        if path.is_file() and path.suffix.lower() in {".md", ".markdown", ".txt", ".pdf"}:
            service.reindex_file(path)
    return IndexResponse(**total.__dict__)


@router.post("/index/cleanup", response_model=CleanupResponse)
def cleanup_index(
    request: IndexPathRequest,
    service: DocumentIndexingService = Depends(get_indexing_service),  # noqa: B008
) -> CleanupResponse:
    return CleanupResponse(removed_sources=service.cleanup_missing_sources(request.path))


@router.post("/search", response_model=SearchResponse)
def search(
    request: SearchRequest,
    service: DocumentIndexingService = Depends(get_indexing_service),  # noqa: B008
) -> SearchResponse:
    """Search the indexed knowledge base and return citation metadata."""

    results = service.search(
        request.query,
        request.top_k,
        source_name=request.source_name,
        file_type=request.file_type,
    )
    return SearchResponse(
        results=[
            SearchResultResponse(
                chunk_id=result.chunk.chunk_id,
                text=result.chunk.text,
                source_name=result.chunk.source_name,
                source_path=result.chunk.source_path,
                file_type=result.chunk.file_type,
                page_number=result.chunk.page_number,
                score=result.score,
            )
            for result in results
        ]
    )


@router.post("/answer", response_model=AnswerResponse)
def answer(
    request: AnswerRequest,
    service: DocumentIndexingService = Depends(get_indexing_service),  # noqa: B008
) -> AnswerResponse:
    """Retrieve evidence and return a citation-aware grounded answer."""

    semantic_weight = _search_mode_weight(request.search_mode, get_settings().retrieval_semantic_weight)
    results = service.search(
        request.query,
        request.top_k,
        source_name=request.source_name,
        file_type=request.file_type,
        semantic_weight=semantic_weight,
    )
    generated = GroundedAnswerService(build_answer_model(get_settings())).generate(
        request.query, results
    )
    threshold = request.confidence_threshold if request.confidence_threshold is not None else get_settings().confidence_warning_threshold
    confidence_warning = bool(generated.citations) and min(citation.confidence for citation in generated.citations) < threshold
    latency_warning = False
    return AnswerResponse(
        answer=generated.answer,
        citations=[CitationResponse(**citation.__dict__) for citation in generated.citations],
        grounded=generated.grounded,
        refusal=generated.refusal,
        confidence_warning=confidence_warning,
        confidence_warning_message="Some evidence has a low retrieval match. Review the cited source carefully." if confidence_warning else None,
        retrieval_diagnostics={"query": request.query, "search_mode": request.search_mode, "top_k": request.top_k, "file_type": request.file_type or "all"},
        latency_warning=latency_warning,
    )


@router.post("/ask", response_model=AskResponse)
def ask(
    request: AnswerRequest,
    service: DocumentIndexingService = Depends(get_indexing_service),  # noqa: B008
) -> AskResponse:
    """Run orchestration, retrieval, generation, and verification together."""

    conversation_id = request.conversation_id or conversation_store.create()
    history = conversation_store.get(conversation_id)
    contextual_query = conversation_store.contextual_query(request.query, history)
    settings = get_settings()
    assistant = KnowledgeAssistantService(
        AgentOrchestrator(service, minimum_evidence_score=settings.minimum_evidence_score, minimum_lexical_overlap=settings.minimum_lexical_overlap, web_search_tool=WebSearchTool(timeout=settings.web_search_timeout_seconds, allowed_domains={domain.strip() for domain in settings.web_search_allowed_domains.split(",") if domain.strip()}) if settings.web_search_enabled and request.web_search else None),
        GroundedAnswerService(build_answer_model(get_settings())),
    )
    semantic_weight = _search_mode_weight(request.search_mode, settings.retrieval_semantic_weight)
    response = assistant.ask(contextual_query, top_k=request.top_k, semantic_weight=semantic_weight, file_type=request.file_type)
    latency_repository.record(response.duration_ms, request.latency_profile, response.stage_timings)
    log_event("assistant_request_completed", duration_ms=response.duration_ms, verified=response.verified, refusal=response.refusal, attempts=response.attempts, latency_profile=request.latency_profile)
    conversation_store.append(conversation_id, request.query, response.answer)
    threshold = request.confidence_threshold if request.confidence_threshold is not None else settings.confidence_warning_threshold
    confidence_warning = bool(response.citations) and min(citation.confidence for citation in response.citations) < threshold
    latency_target = _latency_target(request.latency_profile, settings.response_latency_warning_ms)
    latency_warning = response.duration_ms >= latency_target
    return AskResponse(
        answer=response.answer,
        citations=[CitationResponse(**citation.__dict__) for citation in response.citations],
        grounded=response.grounded,
        refusal=response.refusal,
        verified=response.verified,
        verification_reason=response.verification_reason,
        action=response.action.value,
        attempts=response.attempts,
        duration_ms=response.duration_ms,
        conversation_id=conversation_id,
        confidence_warning=confidence_warning,
        confidence_warning_message="Some evidence has a low retrieval match. Review the cited source carefully." if confidence_warning else None,
        retrieval_diagnostics={"query": contextual_query, "search_mode": request.search_mode, "top_k": request.top_k, "file_type": request.file_type or "all"},
        latency_warning=latency_warning,
        latency_warning_message=f"This response took {response.duration_ms} ms, above the {latency_target:.0f} ms {request.latency_profile} target." if latency_warning else None,
        agent_trace=[
            {"stage": "Planning", "status": "completed", "detail": response.action.value},
            {"stage": "Retrieval", "status": "completed" if response.citations else "no_evidence", "detail": f"{len(response.citations)} source(s), {response.attempts} attempt(s), {response.stage_timings['retrieval']} ms"},
            {"stage": "Generation", "status": "refused" if response.refusal else "completed", "detail": f"{'Grounded response produced' if not response.refusal else 'Insufficient reliable evidence'} · {response.stage_timings['generation']} ms"},
            {"stage": "Verification", "status": "completed" if response.verified else "review", "detail": f"{response.verification_reason} · {response.stage_timings['verification']} ms"},
        ],
    )


@router.get("/metrics/latency-history", response_model=list[LatencyHistoryResponse])
def latency_history() -> list[LatencyHistoryResponse]:
    """Return recent response timing samples."""

    return [LatencyHistoryResponse(**sample) for sample in latency_repository.recent()]


@router.get("/metrics/latency-summary", response_model=LatencySummaryResponse)
def latency_summary() -> LatencySummaryResponse:
    """Return summary statistics for recent response timings."""

    values = sorted(float(sample["duration_ms"]) for sample in latency_repository.recent(100))
    if not values:
        return LatencySummaryResponse(sample_count=0, fastest_ms=None, slowest_ms=None, average_ms=None, p95_ms=None)
    p95_index = min(len(values) - 1, max(0, math.ceil(len(values) * 0.95) - 1))
    samples = latency_repository.recent(100)
    stage_averages = {stage: round(sum(float(sample[f"{stage}_ms"]) for sample in samples if sample[f"{stage}_ms"] is not None) / max(1, sum(sample[f"{stage}_ms"] is not None for sample in samples)), 2) for stage in ("retrieval", "generation", "verification")}
    bottleneck = max(stage_averages, key=stage_averages.get)
    recommendations = {"retrieval": "Retrieval is the main bottleneck. Try fewer sources or a file-type filter to reduce search work.", "generation": "Generation is the main bottleneck. Try the Local latency profile or a smaller local model.", "verification": "Verification is the main bottleneck. Review verification logs and check for unusually large evidence sets."}
    return LatencySummaryResponse(sample_count=len(values), fastest_ms=round(values[0], 2), slowest_ms=round(values[-1], 2), average_ms=round(sum(values) / len(values), 2), p95_ms=round(values[p95_index], 2), stage_averages_ms=stage_averages, recommendation=recommendations[bottleneck])


@router.post("/feedback")
def submit_feedback(request: FeedbackRequest) -> dict[str, str]:
    """Record lightweight answer-quality feedback for local observability."""

    metrics.increment(f"answer_feedback_{request.rating}_total")
    feedback_repository.record(request.rating, request.conversation_id)
    return {"status": "recorded", "rating": request.rating}


@router.get("/feedback/summary")
def feedback_summary() -> dict[str, int]:
    """Return persisted answer-quality feedback totals."""

    return feedback_repository.summary()


@router.post("/evaluation/run", response_model=EvaluationReportResponse)
def run_evaluation(
    request: EvaluationRunRequest | None = None,
    service: DocumentIndexingService = Depends(get_indexing_service),  # noqa: B008
) -> EvaluationReportResponse:
    """Run a selected checked-in evaluation dataset."""

    settings = get_settings()
    dataset = request.dataset if request else "sample"
    if dataset == "multi_document":
        with TemporaryDirectory(prefix="atlas-evaluation-") as temp_dir:
            evaluation_service = DocumentIndexingService(
                Path(temp_dir) / "vectors",
                embedding_provider=service.embedding_provider,
                semantic_weight=settings.retrieval_semantic_weight,
            )
            for document in sorted(Path("evaluation/fixtures").iterdir()):
                evaluation_service.index_file(document)
            assistant = KnowledgeAssistantService(
                AgentOrchestrator(evaluation_service, minimum_evidence_score=settings.minimum_evidence_score, minimum_lexical_overlap=settings.minimum_lexical_overlap),
                GroundedAnswerService(build_answer_model(settings)),
            )
            cases = EvaluationService.load_cases(Path("evaluation/multi_document_cases.json"))
            report = EvaluationService(assistant).run(cases)
            values = report.as_dict()
            _add_evaluation_thresholds(values, settings, request)
            evaluation_repository.record(dataset, values)
            return EvaluationReportResponse(**values)
    assistant = KnowledgeAssistantService(
        AgentOrchestrator(service, minimum_evidence_score=settings.minimum_evidence_score, minimum_lexical_overlap=settings.minimum_lexical_overlap),
        GroundedAnswerService(build_answer_model(settings)),
    )
    cases = EvaluationService.load_cases(Path("evaluation/sample_cases.json"))
    report = EvaluationService(assistant).run(cases)
    values = report.as_dict()
    _add_evaluation_thresholds(values, settings, request)
    evaluation_repository.record(dataset, values)
    return EvaluationReportResponse(**values)


def _add_evaluation_thresholds(values: dict[str, object], settings: Settings, request: EvaluationRunRequest | None) -> None:
    """Annotate a report with explicit quality gates for the UI and API."""

    thresholds = {
        "answerable_retrieval_rate": request.retrieval_threshold if request and request.retrieval_threshold is not None else settings.evaluation_retrieval_threshold,
        "answerable_citation_accuracy": request.citation_threshold if request and request.citation_threshold is not None else settings.evaluation_citation_threshold,
        "answer_match_rate": request.answer_threshold if request and request.answer_threshold is not None else settings.evaluation_regression_threshold,
        "refusal_case_accuracy": request.refusal_threshold if request and request.refusal_threshold is not None else settings.evaluation_regression_threshold,
    }
    issues = [
        f"{metric.replace('_', ' ').title()} is below {threshold:.0%}."
        for metric, threshold in thresholds.items()
        if float(values[metric]) < threshold
    ]
    values["quality_thresholds"] = thresholds
    values["quality_issues"] = issues
    values["quality_passed"] = not issues


@router.get("/evaluation/history", response_model=list[EvaluationHistoryResponse])
def evaluation_history(dataset: str | None = None) -> list[EvaluationHistoryResponse]:
    """Return recent persisted quality-check runs."""

    return [EvaluationHistoryResponse(**run) for run in evaluation_repository.recent(dataset)]


@router.post("/evaluation/compare-providers", response_model=ProviderComparisonResponse)
def compare_providers(settings: Settings = Depends(get_settings)) -> ProviderComparisonResponse:  # noqa: B008
    """Run the sample benchmark with deterministic and Ollama answer generation."""

    ollama_available = False
    try:
        ollama_available = httpx.get(f"{settings.llm_base_url.rstrip('/')}/api/tags", timeout=2.0).is_success
    except httpx.HTTPError:
        pass
    with TemporaryDirectory(prefix="atlas-provider-compare-") as temp_dir:
        indexing = DocumentIndexingService(Path(temp_dir) / "vectors", embedding_provider=LocalHashEmbeddingProvider())
        indexing.index_file(Path("evaluation/fixtures/policy.md") if Path("evaluation/fixtures/policy.md").is_file() else Path("tests/fixtures/policy.md"))
        cases = EvaluationService.load_cases(Path("evaluation/sample_cases.json"))
        orchestrator = AgentOrchestrator(indexing, minimum_evidence_score=settings.minimum_evidence_score, minimum_lexical_overlap=settings.minimum_lexical_overlap)
        deterministic = EvaluationService(KnowledgeAssistantService(orchestrator, GroundedAnswerService())).run(cases).as_dict()
        ollama = None
        if ollama_available:
            ollama_model = OllamaAnswerModel(settings.llm_model, settings.llm_base_url, max_retries=settings.llm_max_retries)
            ollama = EvaluationService(KnowledgeAssistantService(orchestrator, GroundedAnswerService(ollama_model))).run(cases).as_dict()
    thresholds = {"retrieval_hit_rate": settings.evaluation_retrieval_threshold, "answer_match_rate": settings.evaluation_regression_threshold, "citation_accuracy": settings.evaluation_citation_threshold, "refusal_accuracy": settings.evaluation_regression_threshold}
    metrics = tuple(thresholds)
    regression_details = [f"Deterministic {metric} is below {thresholds[metric]:.0%}." for metric in metrics if deterministic[metric] < thresholds[metric]]
    if ollama:
        regression_details.extend(f"Ollama {metric} is below {thresholds[metric]:.0%}." for metric in metrics if ollama[metric] < thresholds[metric])
    return ProviderComparisonResponse(deterministic=deterministic, ollama=ollama, ollama_available=ollama_available, detail="Comparison completed." if ollama_available else "Ollama is unavailable; deterministic results only.", regression_detected=bool(regression_details), regression_details=regression_details)


@router.get("/conversations", response_model=ConversationListResponse)
def list_conversations() -> ConversationListResponse:
    """List persisted conversation IDs."""

    return ConversationListResponse(conversation_ids=conversation_store.list_ids())


@router.get("/conversations/{conversation_id}", response_model=ConversationResponse)
def get_conversation(conversation_id: str) -> ConversationResponse:
    """Return one persisted conversation history."""

    turns = conversation_store.get(conversation_id)
    if not turns and conversation_id not in conversation_store.list_ids():
        raise FileNotFoundError(f"Conversation not found: {conversation_id}")
    return ConversationResponse(
        conversation_id=conversation_id,
        turns=[ConversationTurnResponse(**turn.__dict__) for turn in turns],
    )


@router.delete("/conversations/{conversation_id}")
def delete_conversation(conversation_id: str) -> dict[str, str]:
    """Delete one persisted conversation."""

    if not conversation_store.delete(conversation_id):
        raise FileNotFoundError(f"Conversation not found: {conversation_id}")
    return {"status": "deleted", "conversation_id": conversation_id}

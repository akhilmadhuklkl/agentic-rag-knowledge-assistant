"""Knowledge-base API routes."""

import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi import APIRouter, Depends, File, Query, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse

from agentic_rag.agents.orchestrator import AgentOrchestrator
from agentic_rag.api.schemas import (
    AnswerRequest,
    AnswerResponse,
    AskResponse,
    CitationResponse,
    CleanupResponse,
    ConversationListResponse,
    ConversationResponse,
    ConversationTurnResponse,
    DocumentListResponse,
    DocumentResponse,
    EvaluationHistoryResponse,
    EvaluationReportResponse,
    EvaluationRunRequest,
    FeedbackRequest,
    IndexPathRequest,
    IndexResponse,
    IngestionActivityResponse,
    KnowledgeBaseHealthResponse,
    ReindexStaleResponse,
    SearchRequest,
    SearchResponse,
    SearchResultResponse,
)
from agentic_rag.config import Settings, get_settings
from agentic_rag.conversations.repository import SQLiteConversationRepository
from agentic_rag.conversations.store import ConversationStore
from agentic_rag.embeddings.providers import build_embedding_provider
from agentic_rag.evaluation.service import EvaluationService
from agentic_rag.generation.providers import build_answer_model
from agentic_rag.generation.service import GroundedAnswerService
from agentic_rag.indexing.service import DocumentIndexingService
from agentic_rag.metrics import metrics
from agentic_rag.storage.documents import DocumentStorageService
from agentic_rag.storage.repository import (
    SQLiteDocumentRepository,
    SQLiteEvaluationRepository,
    SQLiteFeedbackRepository,
    SQLiteIngestionRepository,
)
from agentic_rag.workflow.service import KnowledgeAssistantService

router = APIRouter(prefix="/knowledge", tags=["knowledge"])
conversation_store = ConversationStore(
    repository=SQLiteConversationRepository(get_settings().database_path)
)
document_repository = SQLiteDocumentRepository(get_settings().database_path)
feedback_repository = SQLiteFeedbackRepository(get_settings().database_path)
evaluation_repository = SQLiteEvaluationRepository(get_settings().database_path)
ingestion_repository = SQLiteIngestionRepository(get_settings().database_path)


def _search_mode_weight(mode: str, hybrid_weight: float) -> float:
    """Translate a user-facing retrieval mode into a ranking weight."""

    return {"semantic": 1.0, "keyword": 0.0}.get(mode, hybrid_weight)


def _latency_target(profile: str, default: float) -> float:
    """Translate a performance profile into a warning target."""

    return {"local": default * 1.5, "strict": default * 0.6}.get(profile, default)


def get_indexing_service(settings: Settings = Depends(get_settings)) -> DocumentIndexingService:  # noqa: B008
    """Build the application service from runtime configuration."""

    return DocumentIndexingService(
        settings.vector_store_path,
        embedding_provider=build_embedding_provider(settings),
        semantic_weight=settings.retrieval_semantic_weight,
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
        AgentOrchestrator(service, minimum_evidence_score=settings.minimum_evidence_score),
        GroundedAnswerService(build_answer_model(get_settings())),
    )
    semantic_weight = _search_mode_weight(request.search_mode, settings.retrieval_semantic_weight)
    response = assistant.ask(contextual_query, top_k=request.top_k, semantic_weight=semantic_weight, file_type=request.file_type)
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
                AgentOrchestrator(evaluation_service, minimum_evidence_score=settings.minimum_evidence_score),
                GroundedAnswerService(build_answer_model(settings)),
            )
            cases = EvaluationService.load_cases(Path("evaluation/multi_document_cases.json"))
            report = EvaluationService(assistant).run(cases)
            values = report.as_dict()
            evaluation_repository.record(dataset, values)
            return EvaluationReportResponse(**values)
    assistant = KnowledgeAssistantService(
        AgentOrchestrator(service, minimum_evidence_score=settings.minimum_evidence_score),
        GroundedAnswerService(build_answer_model(settings)),
    )
    cases = EvaluationService.load_cases(Path("evaluation/sample_cases.json"))
    report = EvaluationService(assistant).run(cases)
    values = report.as_dict()
    evaluation_repository.record(dataset, values)
    return EvaluationReportResponse(**values)


@router.get("/evaluation/history", response_model=list[EvaluationHistoryResponse])
def evaluation_history(dataset: str | None = None) -> list[EvaluationHistoryResponse]:
    """Return recent persisted quality-check runs."""

    return [EvaluationHistoryResponse(**run) for run in evaluation_repository.recent(dataset)]


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

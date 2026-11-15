"""Knowledge-base API routes."""

from pathlib import Path

from fastapi import APIRouter, Depends, File, UploadFile

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
    FeedbackRequest,
    IndexPathRequest,
    IndexResponse,
    SearchRequest,
    SearchResponse,
    SearchResultResponse,
)
from agentic_rag.config import Settings, get_settings
from agentic_rag.conversations.repository import SQLiteConversationRepository
from agentic_rag.conversations.store import ConversationStore
from agentic_rag.embeddings.providers import build_embedding_provider
from agentic_rag.generation.providers import build_answer_model
from agentic_rag.generation.service import GroundedAnswerService
from agentic_rag.indexing.service import DocumentIndexingService
from agentic_rag.metrics import metrics
from agentic_rag.storage.documents import DocumentStorageService
from agentic_rag.storage.repository import SQLiteDocumentRepository, SQLiteFeedbackRepository
from agentic_rag.workflow.service import KnowledgeAssistantService

router = APIRouter(prefix="/knowledge", tags=["knowledge"])
conversation_store = ConversationStore(
    repository=SQLiteConversationRepository(get_settings().database_path)
)
document_repository = SQLiteDocumentRepository(get_settings().database_path)
feedback_repository = SQLiteFeedbackRepository(get_settings().database_path)


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
) -> DocumentListResponse:
    """List supported documents currently stored locally."""

    return DocumentListResponse(
        documents=[DocumentResponse(**document.__dict__) for document in storage.list_documents()]
    )


@router.post("/documents/upload", response_model=DocumentResponse)
async def upload_document(
    file: UploadFile = File(...),  # noqa: B008
    storage: DocumentStorageService = Depends(get_document_storage),  # noqa: B008
    indexing: DocumentIndexingService = Depends(get_indexing_service),  # noqa: B008
) -> DocumentResponse:
    """Upload one supported document into local controlled storage."""

    document = storage.save(file.filename or "", await file.read())
    indexing.index_file(document.path)
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
    return DocumentResponse(**document.__dict__)


@router.post("/index/file", response_model=IndexResponse)
def index_file(
    request: IndexPathRequest,
    service: DocumentIndexingService = Depends(get_indexing_service),  # noqa: B008
) -> IndexResponse:
    """Index one local document."""

    summary = service.index_file(request.path)
    return IndexResponse(**summary.__dict__)


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

    results = service.search(
        request.query,
        request.top_k,
        source_name=request.source_name,
        file_type=request.file_type,
    )
    generated = GroundedAnswerService(build_answer_model(get_settings())).generate(
        request.query, results
    )
    return AnswerResponse(
        answer=generated.answer,
        citations=[CitationResponse(**citation.__dict__) for citation in generated.citations],
        grounded=generated.grounded,
        refusal=generated.refusal,
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
    assistant = KnowledgeAssistantService(
        AgentOrchestrator(service),
        GroundedAnswerService(build_answer_model(get_settings())),
    )
    response = assistant.ask(contextual_query, top_k=request.top_k)
    conversation_store.append(conversation_id, request.query, response.answer)
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

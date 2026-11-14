"""Knowledge-base API routes."""

from fastapi import APIRouter, Depends, File, UploadFile

from agentic_rag.agents.orchestrator import AgentOrchestrator
from agentic_rag.api.schemas import (
    AnswerRequest,
    AnswerResponse,
    AskResponse,
    CitationResponse,
    DocumentListResponse,
    DocumentResponse,
    IndexPathRequest,
    IndexResponse,
    SearchRequest,
    SearchResponse,
    SearchResultResponse,
)
from agentic_rag.config import Settings, get_settings
from agentic_rag.generation.providers import build_answer_model
from agentic_rag.generation.service import GroundedAnswerService
from agentic_rag.indexing.service import DocumentIndexingService
from agentic_rag.storage.documents import DocumentStorageService
from agentic_rag.workflow.service import KnowledgeAssistantService

router = APIRouter(prefix="/knowledge", tags=["knowledge"])


def get_indexing_service(settings: Settings = Depends(get_settings)) -> DocumentIndexingService:  # noqa: B008
    """Build the application service from runtime configuration."""

    return DocumentIndexingService(
        settings.vector_store_path,
        semantic_weight=settings.retrieval_semantic_weight,
    )


def get_document_storage(settings: Settings = Depends(get_settings)) -> DocumentStorageService:  # noqa: B008
    """Build the controlled local document storage service."""

    return DocumentStorageService(settings.document_storage_path)


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

    assistant = KnowledgeAssistantService(
        AgentOrchestrator(service),
        GroundedAnswerService(build_answer_model(get_settings())),
    )
    response = assistant.ask(request.query, top_k=request.top_k)
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
    )

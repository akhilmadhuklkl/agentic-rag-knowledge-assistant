"""API request and response models."""

from typing import Literal

from pydantic import BaseModel, Field


class IndexPathRequest(BaseModel):
    """Request to index a local file or directory path."""

    path: str = Field(min_length=1, max_length=1000)


class DocumentResponse(BaseModel):
    """Stored document metadata."""

    name: str
    path: str
    size_bytes: int
    file_type: str
    status: Literal["indexed", "stale", "missing", "unknown"] = "unknown"
    upload_date: str | None = None


class DocumentListResponse(BaseModel):
    """Available stored documents."""

    documents: list[DocumentResponse]


class KnowledgeBaseHealthResponse(BaseModel):
    """Summary of document and index readiness."""

    document_count: int
    indexed_document_count: int
    chunk_count: int
    stale_document_count: int
    status: Literal["ready", "needs_reindex", "empty"]


class ReindexStaleResponse(BaseModel):
    """Result of refreshing stale knowledge-base sources."""

    reindexed_count: int
    skipped_missing_count: int
    chunk_count: int
    processed_documents: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class IngestionActivityResponse(BaseModel):
    """One document ingestion event."""

    action: str
    document_name: str
    detail: str
    created_at: str


class LatencyHistoryResponse(BaseModel):
    duration_ms: float
    profile: str
    created_at: str


class LatencySummaryResponse(BaseModel):
    sample_count: int
    fastest_ms: float | None
    slowest_ms: float | None
    average_ms: float | None
    p95_ms: float | None


class CleanupResponse(BaseModel):
    removed_sources: int


class IndexResponse(BaseModel):
    """Summary returned after indexing."""

    source_count: int
    chunk_count: int
    vector_store_path: str
    indexed_count: int
    skipped_count: int


class SearchRequest(BaseModel):
    """Search request for the indexed knowledge base."""

    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=50)
    source_name: str | None = Field(default=None, min_length=1)
    file_type: str | None = Field(default=None, min_length=1)
    search_mode: Literal["hybrid", "semantic", "keyword"] = "hybrid"


class SearchResultResponse(BaseModel):
    """Citation-ready search result."""

    chunk_id: str
    text: str
    source_name: str
    source_path: str
    file_type: str
    page_number: int | None
    score: float


class SearchResponse(BaseModel):
    """Search response containing ranked evidence."""

    results: list[SearchResultResponse]


class AnswerRequest(SearchRequest):
    """Request to retrieve evidence and generate a grounded answer."""

    conversation_id: str | None = Field(default=None, max_length=64)
    confidence_threshold: float | None = Field(default=None, ge=0, le=1)
    latency_profile: Literal["local", "balanced", "strict"] = "balanced"


class FeedbackRequest(BaseModel):
    """User quality feedback for one generated answer."""

    rating: Literal["helpful", "needs_improvement"]
    conversation_id: str | None = Field(default=None, max_length=64)


class EvaluationRunRequest(BaseModel):
    """Select the isolated benchmark corpus to evaluate."""

    dataset: Literal["sample", "multi_document"] = "sample"


class EvaluationCaseResponse(BaseModel):
    """Result details for one quality benchmark case."""

    question: str
    expected_source: str | None
    cited_sources: list[str]
    retrieval_hit: bool
    answer_match: bool
    citation_hit: bool
    refusal_correct: bool


class EvaluationReportResponse(BaseModel):
    """Quality report from the repeatable evaluation dataset."""

    total_cases: int
    retrieval_hits: int
    answer_matches: int
    citation_hits: int
    correct_refusals: int
    retrieval_hit_rate: float
    answer_match_rate: float
    citation_accuracy: float
    refusal_accuracy: float
    cases: list[EvaluationCaseResponse]


class EvaluationHistoryResponse(BaseModel):
    """A compact historical benchmark result."""

    dataset: str
    total_cases: int
    retrieval_hit_rate: float
    answer_match_rate: float
    citation_accuracy: float
    refusal_accuracy: float
    created_at: str


class CitationResponse(BaseModel):
    """Citation returned with a grounded answer."""

    citation_id: int
    chunk_id: str
    source_name: str
    page_number: int | None
    excerpt: str
    confidence: float
    semantic_score: float
    lexical_score: float


class AnswerResponse(BaseModel):
    """Grounded answer response."""

    answer: str
    citations: list[CitationResponse]
    grounded: bool
    refusal: bool
    confidence_warning: bool = False
    confidence_warning_message: str | None = None
    retrieval_diagnostics: dict[str, object] = Field(default_factory=dict)
    latency_warning: bool = False
    latency_warning_message: str | None = None


class AskResponse(AnswerResponse):
    """Complete assistant workflow response."""

    verified: bool
    verification_reason: str
    action: str
    attempts: int
    duration_ms: float
    conversation_id: str
    agent_trace: list[dict[str, object]] = Field(default_factory=list)


class ConversationTurnResponse(BaseModel):
    """One persisted conversation turn."""

    question: str
    answer: str


class ConversationResponse(BaseModel):
    """Conversation history response."""

    conversation_id: str
    turns: list[ConversationTurnResponse]


class ConversationListResponse(BaseModel):
    """Known conversation IDs."""

    conversation_ids: list[str]

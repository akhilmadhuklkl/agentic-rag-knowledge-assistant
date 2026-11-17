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
    authority: str = "standard"
    freshness: str = "unspecified"
    status: Literal["indexed", "stale", "missing", "unknown"] = "unknown"
    upload_date: str | None = None


class DocumentMetadataUpdate(BaseModel):
    """Editable authority and freshness labels for a stored document."""

    authority: Literal["authoritative", "standard", "legacy"]
    freshness: Literal["current", "historical", "unspecified"]


class BulkDocumentMetadataUpdate(BaseModel):
    """Apply labels to several stored documents."""

    names: list[str] = Field(min_length=1, max_length=100)
    authority: Literal["authoritative", "standard", "legacy"]
    freshness: Literal["current", "historical", "unspecified"]


class BulkDocumentMetadataResponse(BaseModel):
    """Per-document result summary for a batch metadata update."""

    updated: list[DocumentResponse] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class BulkDocumentDeleteRequest(BaseModel):
    """Select documents to remove from storage and the knowledge index."""

    names: list[str] = Field(min_length=1, max_length=100)


class BulkDocumentDeleteResponse(BaseModel):
    """Per-document result summary for a batch deletion."""

    deleted: list[DocumentResponse] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class BulkDocumentMetadataUndo(BaseModel):
    """Select documents whose latest metadata edit should be undone."""

    names: list[str] = Field(min_length=1, max_length=100)


class DocumentListResponse(BaseModel):
    """Available stored documents."""

    documents: list[DocumentResponse]


class RecycleBinItemResponse(BaseModel):
    name: str
    deleted_at: str
    age_days: int


class RecycleBinCleanupResponse(BaseModel):
    removed: list[str] = Field(default_factory=list)


class KnowledgeBaseHealthResponse(BaseModel):
    """Summary of document and index readiness."""

    document_count: int
    indexed_document_count: int
    chunk_count: int
    stale_document_count: int
    status: Literal["ready", "needs_reindex", "empty"]


class ProviderHealthResponse(BaseModel):
    status: Literal["ready", "degraded"]
    llm_provider: str
    llm_model: str
    embedding_provider: str
    embedding_model: str
    ollama_reachable: bool
    detail: str
    embedding_provider_id: str = "unknown"
    indexed_provider_id: str | None = None
    index_status: Literal["compatible", "needs_reindex", "not_created"] = "not_created"


class SecurityAuditEventResponse(BaseModel):
    request_id: str
    method: str
    path: str
    reason: str
    severity: Literal["low", "medium", "high"]
    created_at: str


class AdminSettingsResponse(BaseModel):
    audit_retention_days: int
    recycle_bin_retention_days: int
    configured_admin_count: int
    admin_configuration: str


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
    stage_averages_ms: dict[str, float] = Field(default_factory=dict)
    recommendation: str | None = None


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
    web_search: bool = False


class FeedbackRequest(BaseModel):
    """User quality feedback for one generated answer."""

    rating: Literal["helpful", "needs_improvement"]
    conversation_id: str | None = Field(default=None, max_length=64)


class EvaluationRunRequest(BaseModel):
    """Select the isolated benchmark corpus to evaluate."""

    dataset: Literal["sample", "multi_document"] = "sample"
    retrieval_threshold: float | None = Field(default=None, ge=0, le=1)
    citation_threshold: float | None = Field(default=None, ge=0, le=1)
    answer_threshold: float | None = Field(default=None, ge=0, le=1)
    refusal_threshold: float | None = Field(default=None, ge=0, le=1)


class EvaluationCaseResponse(BaseModel):
    """Result details for one quality benchmark case."""

    question: str
    category: str = "general"
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
    answerable_cases: int = 0
    refusal_cases: int = 0
    answerable_retrieval_rate: float = 0.0
    answerable_citation_accuracy: float = 0.0
    refusal_case_accuracy: float = 0.0
    refusal_case_correct: int = 0
    category_metrics: dict[str, dict[str, int | float]] = Field(default_factory=dict)
    quality_passed: bool = True
    quality_thresholds: dict[str, float] = Field(default_factory=dict)
    quality_issues: list[str] = Field(default_factory=list)


class ProviderComparisonResponse(BaseModel):
    """Quality comparison for two answer-provider configurations."""

    deterministic: dict[str, object]
    ollama: dict[str, object] | None = None
    ollama_available: bool
    detail: str
    regression_detected: bool = False
    regression_details: list[str] = Field(default_factory=list)


class EvaluationHistoryResponse(BaseModel):
    """A compact historical benchmark result."""

    dataset: str
    total_cases: int
    retrieval_hit_rate: float
    answer_match_rate: float
    citation_accuracy: float
    refusal_accuracy: float
    answerable_retrieval_rate: float = 0.0
    answerable_citation_accuracy: float = 0.0
    refusal_case_accuracy: float = 0.0
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
    proximity_score: float = 0.0
    authority: str = "standard"
    freshness: str = "unspecified"
    file_type: str = "document"
    source_url: str | None = None
    source_quality: float = 0.0


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

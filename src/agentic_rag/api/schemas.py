"""API request and response models."""

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


class DocumentListResponse(BaseModel):
    """Available stored documents."""

    documents: list[DocumentResponse]


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


class CitationResponse(BaseModel):
    """Citation returned with a grounded answer."""

    citation_id: int
    chunk_id: str
    source_name: str
    page_number: int | None


class AnswerResponse(BaseModel):
    """Grounded answer response."""

    answer: str
    citations: list[CitationResponse]
    grounded: bool
    refusal: bool


class AskResponse(AnswerResponse):
    """Complete assistant workflow response."""

    verified: bool
    verification_reason: str
    action: str
    attempts: int
    duration_ms: float
    conversation_id: str


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

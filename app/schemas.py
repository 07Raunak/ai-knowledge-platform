from datetime import datetime, timezone
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, PlainSerializer, field_validator

FileType = Literal["pdf", "markdown", "text", "code"]


def _utc_iso(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    if dt.tzinfo is None:  # SQLite returns naive datetimes; they are stored as UTC
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat().replace("+00:00", "Z")


UTCDateTime = Annotated[datetime, PlainSerializer(_utc_iso, return_type=str | None)]


# --------------------------------------------------------------------- documents
class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    filename: str
    file_type: FileType
    language: str | None
    size_bytes: int
    sha256: str
    status: Literal["queued", "processing", "ready", "failed", "deleted"]
    error: str | None
    attempts: int
    chunk_count: int
    page_count: int | None
    embedding_model: str | None
    tags: list[str]
    metadata: dict
    uploaded_by: str
    created_at: UTCDateTime
    updated_at: UTCDateTime
    processed_at: UTCDateTime | None
    deleted_at: UTCDateTime | None

    @classmethod
    def from_model(cls, doc) -> "DocumentOut":
        return cls(
            id=doc.id,
            filename=doc.filename,
            file_type=doc.file_type,
            language=doc.language,
            size_bytes=doc.size_bytes,
            sha256=doc.sha256,
            status=doc.status,
            error=doc.error,
            attempts=doc.attempts,
            chunk_count=doc.chunk_count,
            page_count=doc.page_count,
            embedding_model=doc.embedding_model,
            tags=doc.tag_names,
            metadata=doc.extra_metadata or {},
            uploaded_by=doc.uploaded_by,
            created_at=doc.created_at,
            updated_at=doc.updated_at,
            processed_at=doc.processed_at,
            deleted_at=doc.deleted_at,
        )


class UploadResponse(BaseModel):
    document: DocumentOut
    duplicate: bool = Field(description="True if identical content already existed; no re-processing was queued")
    status_url: str


class DocumentList(BaseModel):
    items: list[DocumentOut]
    total: int
    limit: int
    offset: int


class ChunkOut(BaseModel):
    id: str
    chunk_index: int
    content: str
    token_count: int
    chunk_type: str
    section: str | None
    page_start: int | None
    page_end: int | None
    start_line: int | None
    end_line: int | None


class ChunkList(BaseModel):
    document_id: str
    items: list[ChunkOut]
    total: int
    limit: int
    offset: int


class DeleteResponse(BaseModel):
    id: str
    status: Literal["deleted"]
    purged: bool = Field(description="True once metadata, chunks and vectors are physically removed")
    message: str


# ------------------------------------------------------------------------- query
class QueryFilters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_ids: list[str] | None = None
    file_types: list[FileType] | None = None
    languages: list[str] | None = Field(None, description="Code language, e.g. ['python']")
    tags: list[str] | None = Field(None, description="Match documents having ANY of these tags")
    metadata: dict[str, str | int | float | bool] | None = Field(
        None, description="Exact-match on custom upload metadata, e.g. {'team': 'platform'}"
    )
    uploaded_by: str | None = None
    filename_contains: str | None = None
    created_after: datetime | None = None
    created_before: datetime | None = None

    @field_validator("metadata")
    @classmethod
    def _keys(cls, v):
        if v:
            for key in v:
                if not key.replace("_", "").replace("-", "").isalnum() or len(key) > 64:
                    raise ValueError(f"invalid metadata key: {key!r}")
        return v

    def is_empty(self) -> bool:
        return not any(v not in (None, [], {}) for v in self.model_dump().values())


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(5, ge=1, le=50)
    filters: QueryFilters | None = None
    rerank: bool | None = Field(None, description="Override server default for cross-encoder reranking")
    hybrid: bool | None = Field(None, description="Override server default for vector+keyword fusion")
    min_score: float | None = Field(None, ge=-1, le=1)
    generate_answer: bool = Field(False, description="Also generate a grounded answer via the AI Gateway")

    @field_validator("query")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("query must not be blank")
        return v


class ScoreBreakdown(BaseModel):
    vector_similarity: float | None
    vector_rank: int | None
    keyword_rank: int | None
    rrf: float
    rerank: float | None


class QueryResult(BaseModel):
    rank: int
    score: float
    chunk_id: str
    document_id: str
    filename: str
    file_type: str
    language: str | None
    chunk_index: int
    section: str | None
    page_start: int | None
    page_end: int | None
    start_line: int | None
    end_line: int | None
    content: str
    scores: ScoreBreakdown


class Answer(BaseModel):
    text: str | None
    model: str | None
    citations: list[int] = Field(default_factory=list, description="1-based ranks of cited results")
    error: str | None = None


class QueryResponse(BaseModel):
    query_id: str
    query: str
    results: list[QueryResult]
    answer: Answer | None = None
    reranked: bool
    hybrid: bool
    cache_hit: bool
    timings_ms: dict[str, float]


# ----------------------------------------------------------------------- gateway
class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1)


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    messages: list[ChatMessage] = Field(..., min_length=1)
    system: str | None = None
    model: str | None = None
    max_tokens: int | None = Field(None, ge=1, le=32000)


class ChatResponse(BaseModel):
    model: str
    content: str
    stop_reason: str | None
    input_tokens: int
    output_tokens: int
    latency_ms: float
    fallback_used: bool

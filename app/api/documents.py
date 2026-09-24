import hashlib
import json
import mimetypes
import re
import uuid
from pathlib import PurePath

from fastapi import APIRouter, Depends, File, Form, Query, Response, UploadFile
from sqlalchemy import func, select

from app.api.deps import get_container, get_current_user
from app.container import Container
from app.db.models import Chunk, Document, DocumentStatus, DocumentTag
from app.errors import AppError, NotFoundError, PayloadTooLargeError, UnsupportedFileError
from app.ingestion.filetypes import detect_file_kind, supported_extensions
from app.schemas import ChunkList, ChunkOut, DeleteResponse, DocumentList, DocumentOut, UploadResponse

router = APIRouter(prefix="/v1/documents", tags=["documents"])

_TAG_RE = re.compile(r"^[a-z0-9][a-z0-9_.:-]{0,63}$")


class ValidationAppError(AppError):
    status_code = 422
    code = "validation_error"


def _parse_tags(raw: str | None) -> list[str]:
    if not raw:
        return []
    tags = sorted({t.strip().lower() for t in raw.split(",") if t.strip()})
    bad = [t for t in tags if not _TAG_RE.match(t)]
    if bad:
        raise ValidationAppError("Invalid tag(s)", details={"invalid": bad, "pattern": _TAG_RE.pattern})
    if len(tags) > 20:
        raise ValidationAppError("At most 20 tags are allowed")
    return tags


def _parse_metadata(raw: str | None) -> dict:
    if not raw:
        return {}
    try:
        meta = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValidationAppError(f"metadata must be a JSON object: {exc.msg}") from exc
    if not isinstance(meta, dict) or not all(
        isinstance(v, (str, int, float, bool)) and len(str(k)) <= 64 for k, v in meta.items()
    ):
        raise ValidationAppError("metadata must be a flat JSON object of scalar values")
    return meta


def _get_doc(c: Container, document_id: str) -> Document:
    with c.sessions() as s:
        doc = s.get(Document, document_id)
        if doc is None:
            raise NotFoundError(f"Document {document_id} not found")
        return doc


@router.post(
    "",
    status_code=202,
    response_model=UploadResponse,
    summary="Upload a document or code file for asynchronous ingestion",
    responses={
        200: {"description": "Identical content already ingested; existing document returned"},
        413: {"description": "File too large"},
        415: {"description": "Unsupported file type"},
    },
)
def upload_document(
    response: Response,
    file: UploadFile = File(..., description="PDF, markdown, text or source-code file"),
    tags: str | None = Form(None, description="Comma-separated tags, e.g. 'ai,whitepaper'"),
    metadata: str | None = Form(None, description='Flat JSON object, e.g. {"team": "platform"}'),
    user: str = Depends(get_current_user),
    c: Container = Depends(get_container),
):
    filename = PurePath(file.filename or "").name
    kind = detect_file_kind(filename)
    if kind is None:
        raise UnsupportedFileError(
            f"Unsupported file type for '{filename}'", details={"supported_extensions": supported_extensions()}
        )
    tag_list = _parse_tags(tags)
    meta = _parse_metadata(metadata)

    max_bytes = c.settings.max_upload_mb * 1024 * 1024
    data = file.file.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise PayloadTooLargeError(f"File exceeds the {c.settings.max_upload_mb} MB limit")
    if not data:
        raise ValidationAppError("File is empty")
    if kind.file_type == "pdf" and not data.startswith(b"%PDF-"):
        raise UnsupportedFileError("File has a .pdf extension but is not a PDF")

    sha256 = hashlib.sha256(data).hexdigest()
    doc_id = str(uuid.uuid4())
    try:
        return _create_document(c, response, doc_id, filename, kind, file.content_type, data, sha256, meta, tag_list, user)
    except Exception:
        c.storage.delete_document(doc_id)  # don't leave an orphaned file if the DB write failed
        raise


def _create_document(c, response, doc_id, filename, kind, content_type, data, sha256, meta, tag_list, user):
    with c.sessions.begin() as s:
        # Content-addressed de-duplication: re-uploading identical bytes is a no-op.
        existing = s.execute(
            select(Document).where(
                Document.sha256 == sha256,
                Document.status.not_in([DocumentStatus.DELETED, DocumentStatus.FAILED]),
            )
        ).scalars().first()
        if existing is not None:
            response.status_code = 200
            return UploadResponse(
                document=DocumentOut.from_model(existing),
                duplicate=True,
                status_url=f"/v1/documents/{existing.id}",
            )

        storage_key = c.storage.put(doc_id, filename, data)
        doc = Document(
            id=doc_id,
            filename=filename,
            file_type=kind.file_type,
            language=kind.language,
            mime_type=content_type or mimetypes.guess_type(filename)[0],
            size_bytes=len(data),
            sha256=sha256,
            storage_path=storage_key,
            extra_metadata=meta,
            uploaded_by=user,
            status=DocumentStatus.QUEUED,
            attempts=0,
            purge_attempts=0,
            chunk_count=0,
        )
        doc.tags = [DocumentTag(tag=t) for t in tag_list]
        s.add(doc)
        s.flush()
        out = DocumentOut.from_model(doc)

    c.worker.notify()
    response.headers["Location"] = f"/v1/documents/{doc_id}"
    return UploadResponse(document=out, duplicate=False, status_url=f"/v1/documents/{doc_id}")


@router.get("", response_model=DocumentList, summary="List documents")
def list_documents(
    status: str | None = Query(None, description="queued | processing | ready | failed | deleted"),
    file_type: str | None = Query(None),
    tag: str | None = Query(None),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    _: str = Depends(get_current_user),
    c: Container = Depends(get_container),
):
    stmt = select(Document)
    stmt = stmt.where(Document.status == status) if status else stmt.where(Document.status != DocumentStatus.DELETED)
    if file_type:
        stmt = stmt.where(Document.file_type == file_type)
    if tag:
        stmt = stmt.where(Document.id.in_(select(DocumentTag.document_id).where(DocumentTag.tag == tag.lower())))
    with c.sessions() as s:
        total = s.execute(select(func.count()).select_from(stmt.subquery())).scalar_one()
        docs = s.execute(stmt.order_by(Document.created_at.desc()).limit(limit).offset(offset)).scalars().all()
        return DocumentList(items=[DocumentOut.from_model(d) for d in docs], total=total, limit=limit, offset=offset)


@router.get("/{document_id}", response_model=DocumentOut, summary="Get document metadata and processing status")
def get_document(document_id: str, _: str = Depends(get_current_user), c: Container = Depends(get_container)):
    return DocumentOut.from_model(_get_doc(c, document_id))


@router.get("/{document_id}/chunks", response_model=ChunkList, summary="Inspect a document's chunks")
def list_chunks(
    document_id: str,
    limit: int = Query(20, ge=1, le=200),
    offset: int = Query(0, ge=0),
    _: str = Depends(get_current_user),
    c: Container = Depends(get_container),
):
    doc = _get_doc(c, document_id)
    if doc.status == DocumentStatus.DELETED:
        raise NotFoundError(f"Document {document_id} not found")
    with c.sessions() as s:
        total = s.execute(select(func.count()).where(Chunk.document_id == document_id)).scalar_one()
        rows = s.execute(
            select(Chunk).where(Chunk.document_id == document_id).order_by(Chunk.chunk_index).limit(limit).offset(offset)
        ).scalars()
        items = [
            ChunkOut(
                id=ch.id, chunk_index=ch.chunk_index, content=ch.content, token_count=ch.token_count,
                chunk_type=ch.chunk_type, section=ch.section, page_start=ch.page_start, page_end=ch.page_end,
                start_line=ch.start_line, end_line=ch.end_line,
            )
            for ch in rows
        ]
    return ChunkList(document_id=document_id, items=items, total=total, limit=limit, offset=offset)


@router.delete(
    "/{document_id}",
    status_code=202,
    response_model=DeleteResponse,
    summary="Delete a document (soft delete immediately, hard purge of chunks/vectors after)",
    responses={200: {"description": "hard=true and the purge completed synchronously"}},
)
def delete_document(
    document_id: str,
    response: Response,
    hard: bool = Query(False, description="Attempt the physical purge synchronously"),
    _: str = Depends(get_current_user),
    c: Container = Depends(get_container),
):
    with c.sessions.begin() as s:
        doc = s.get(Document, document_id)
        if doc is None:
            raise NotFoundError(f"Document {document_id} not found")
        if doc.status != DocumentStatus.DELETED:
            # Step 1 (atomic, cannot partially fail): hide the document from every query.
            c.pipeline.soft_delete(s, doc)

    if hard and c.worker.purge_now(document_id):
        response.status_code = 200
        return DeleteResponse(id=document_id, status="deleted", purged=True,
                              message="Document, chunks, embeddings and file permanently removed")
    # Step 2 (idempotent, retried with backoff by the worker): physical purge.
    c.worker.notify()
    return DeleteResponse(
        id=document_id, status="deleted", purged=False,
        message="Document hidden from search immediately; physical purge scheduled",
    )


@router.post(
    "/{document_id}/reindex",
    status_code=202,
    response_model=DocumentOut,
    summary="Re-run extraction, chunking and embedding (e.g. after a model or chunker change)",
)
def reindex_document(document_id: str, _: str = Depends(get_current_user), c: Container = Depends(get_container)):
    with c.sessions.begin() as s:
        doc = s.get(Document, document_id)
        if doc is None or doc.status == DocumentStatus.DELETED:
            raise NotFoundError(f"Document {document_id} not found")
        if doc.status in (DocumentStatus.QUEUED, DocumentStatus.PROCESSING):
            raise AppError("Document is already being processed")
        doc.status = DocumentStatus.QUEUED
        doc.attempts = 0
        doc.error = None
        doc.next_attempt_at = None
        s.flush()
        out = DocumentOut.from_model(doc)
    c.corpus_version.bump()
    c.worker.notify()
    return out

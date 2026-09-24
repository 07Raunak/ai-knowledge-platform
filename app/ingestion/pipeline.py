"""Ingestion and purge pipelines, executed by the background worker.

Consistency model: the relational DB is the source of truth. Vectors are written first,
then chunks + embedding rows + keyword index are committed in ONE transaction together
with the document's transition to ``ready``. If anything fails in between, the retry
re-writes the same deterministic chunk IDs (upsert), and queries never surface vectors
whose chunk rows are missing or whose document is not ``ready``.
"""

import logging
import uuid
from datetime import datetime, timezone

from sqlalchemy import delete, select, update
from sqlalchemy.orm import Session, sessionmaker

from app.config import Settings
from app.db.models import Chunk, Document, DocumentStatus, DocumentTag, Embedding
from app.embeddings import Embedder
from app.errors import ExtractionError
from app.ingestion import chunkers, extractors
from app.retrieval.cache import CorpusVersion
from app.retrieval.keyword import KeywordIndex
from app.storage import LocalFileStorage
from app.vectorstore import VectorStore

log = logging.getLogger(__name__)

_CHUNK_NS = uuid.UUID("6f1c9a36-6a2e-4c1e-9d0a-3c0f4b8e2a11")


def chunk_id_for(document_id: str, index: int) -> str:
    return str(uuid.uuid5(_CHUNK_NS, f"{document_id}:{index}"))


def build_chunks(
    doc: Document, data: bytes, settings: Settings, ocr: extractors.PdfOcr | None = None
) -> tuple[list[chunkers.ChunkDraft], int | None]:
    size, overlap = settings.chunk_size_tokens, settings.chunk_overlap_tokens
    if doc.file_type == "pdf":
        pages = extractors.extract_pdf(data, ocr)
        return chunkers.chunk_pdf(pages, doc.filename, size, overlap), len(pages)
    text = extractors.decode_text(data)
    if not text.strip():
        raise ExtractionError("File is empty")
    if doc.file_type == "markdown":
        return chunkers.chunk_markdown(text, doc.filename, size, overlap), None
    if doc.file_type == "code":
        if doc.language == "python":
            return chunkers.chunk_python(text, doc.filename, size, overlap), None
        return chunkers.chunk_code_generic(text, doc.filename, doc.language or "text", size, overlap), None
    return chunkers.chunk_plain_text(text, doc.filename, size, overlap), None


class IngestionPipeline:
    def __init__(
        self,
        settings: Settings,
        sessions: sessionmaker[Session],
        storage: LocalFileStorage,
        embedder: Embedder,
        vectors: VectorStore,
        keyword: KeywordIndex,
        corpus_version: CorpusVersion,
    ):
        self.settings = settings
        self.sessions = sessions
        self.storage = storage
        self.embedder = embedder
        self.vectors = vectors
        self.keyword = keyword
        self.corpus_version = corpus_version
        self.ocr = (
            extractors.PdfOcr(settings.ocr_dpi, settings.ocr_rescue_dpi) if settings.ocr_enabled else None
        )

    # ------------------------------------------------------------------ ingest
    def ingest(self, document_id: str) -> None:
        with self.sessions() as s:
            doc = s.get(Document, document_id)
            if doc is None or doc.status != DocumentStatus.PROCESSING:
                return
            data = self.storage.get(doc.storage_path)
            drafts, page_count = build_chunks(doc, data, self.settings, self.ocr)
            if not drafts:
                raise ExtractionError("No content could be extracted from the file")

            ids = [chunk_id_for(doc.id, i) for i in range(len(drafts))]
            vectors = self.embedder.embed_documents([d.embed_text for d in drafts])
            base_meta = {"document_id": doc.id, "file_type": doc.file_type, "language": doc.language or ""}
            self.vectors.upsert(ids, vectors, [{**base_meta, "chunk_index": i} for i in range(len(drafts))])

        now = datetime.now(timezone.utc)
        with self.sessions.begin() as s:
            # Re-check state inside the write transaction: the document may have been
            # deleted while we were embedding.
            doc = s.get(Document, document_id)
            if doc is None or doc.status != DocumentStatus.PROCESSING:
                log.info("Document %s changed state during ingestion; discarding vectors", document_id)
                self.vectors.delete_document(document_id)
                return
            self._delete_chunk_rows(s, document_id)  # idempotent re-ingest
            s.add_all(
                Chunk(
                    id=cid,
                    document_id=doc.id,
                    chunk_index=i,
                    content=d.content,
                    token_count=d.token_count,
                    chunk_type=d.chunk_type,
                    section=d.section,
                    page_start=d.page_start,
                    page_end=d.page_end,
                    start_line=d.start_line,
                    end_line=d.end_line,
                    extra_metadata=d.metadata,
                )
                for i, (cid, d) in enumerate(zip(ids, drafts))
            )
            s.flush()
            dim = len(vectors[0])
            s.add_all(
                Embedding(chunk_id=cid, model=self.embedder.model_name, dimension=dim,
                          vector_collection=self.vectors.collection_name)
                for cid in ids
            )
            self.keyword.add(s, [(cid, doc.id, d.content) for cid, d in zip(ids, drafts)])
            doc.status = DocumentStatus.READY
            doc.chunk_count = len(drafts)
            doc.page_count = page_count
            doc.embedding_model = self.embedder.model_name
            doc.processed_at = now
            doc.error = None
            doc.locked_until = None
            doc.next_attempt_at = None
        self.corpus_version.bump()
        log.info("Ingested document %s (%s): %d chunks", document_id, doc.filename, len(drafts))

    def _delete_chunk_rows(self, s: Session, document_id: str) -> None:
        self.keyword.delete_document(s, document_id)
        chunk_ids = select(Chunk.id).where(Chunk.document_id == document_id)
        s.execute(delete(Embedding).where(Embedding.chunk_id.in_(chunk_ids)))
        s.execute(delete(Chunk).where(Chunk.document_id == document_id))

    # ------------------------------------------------------------------- purge
    def purge(self, document_id: str) -> None:
        """Hard delete of a soft-deleted document. Each step is idempotent, so a partial
        failure is simply retried later; until then the document stays invisible."""
        # 1. Vectors (external system, most likely to fail) first.
        self.vectors.delete_document(document_id)
        # 2. Relational rows in one transaction: FTS rows, embeddings, chunks, tags, document.
        with self.sessions.begin() as s:
            doc = s.get(Document, document_id)
            if doc is None:
                return
            if doc.status != DocumentStatus.DELETED:
                return
            self._delete_chunk_rows(s, document_id)
            s.execute(delete(DocumentTag).where(DocumentTag.document_id == document_id))
            s.execute(delete(Document).where(Document.id == document_id))  # idempotent
        # 3. Raw file last - if this fails the orphaned folder is harmless and swept later.
        self.storage.delete_document(document_id)
        log.info("Purged document %s", document_id)

    def soft_delete(self, s: Session, doc: Document) -> None:
        s.execute(
            update(Document)
            .where(Document.id == doc.id)
            .values(
                status=DocumentStatus.DELETED,
                deleted_at=datetime.now(timezone.utc),
                next_attempt_at=None,
                locked_until=None,
                purge_attempts=0,
            )
        )
        self.corpus_version.bump()

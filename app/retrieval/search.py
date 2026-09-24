"""Query-time retrieval pipeline.

    query --embed (LRU cached)--> vector top-N  --+
          --FTS5 bm25---------->  keyword top-N --+--> RRF fusion --> hydrate from SQL
                                                        (drops non-ready / deleted docs)
          --> cross-encoder rerank top-M --> top-K --> (optional) grounded LLM answer

Metadata filters are resolved against the relational ``documents`` table first (it holds
tags / metadata / status) and pushed down into both retrievers as a document-id
allow-list, so filtering happens *before* top-N truncation rather than after it.
"""

import json
import logging
import time
from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app.config import Settings
from app.db.models import Chunk, Document, DocumentStatus, DocumentTag
from app.embeddings import Embedder
from app.retrieval.cache import CorpusVersion, TTLCache
from app.retrieval.keyword import KeywordIndex
from app.retrieval.reranker import CrossEncoderReranker
from app.schemas import QueryFilters
from app.vectorstore import VectorStore

log = logging.getLogger(__name__)


@dataclass
class Candidate:
    chunk_id: str
    vector_similarity: float | None = None
    vector_rank: int | None = None
    keyword_rank: int | None = None
    rrf: float = 0.0
    rerank: float | None = None
    chunk: Chunk | None = None
    document: Document | None = None


@dataclass
class SearchOutcome:
    candidates: list[Candidate]
    reranked: bool
    hybrid: bool
    cache_hit: bool = False
    timings_ms: dict = field(default_factory=dict)


class SearchService:
    def __init__(
        self,
        settings: Settings,
        sessions: sessionmaker[Session],
        embedder: Embedder,
        vectors: VectorStore,
        keyword: KeywordIndex,
        reranker: CrossEncoderReranker | None,
        corpus_version: CorpusVersion,
    ):
        self.settings = settings
        self.sessions = sessions
        self.embedder = embedder
        self.vectors = vectors
        self.keyword = keyword
        self.reranker = reranker
        self.corpus_version = corpus_version
        self._embedding_cache = TTLCache(settings.query_cache_size, ttl_s=24 * 3600)
        self._result_cache = TTLCache(settings.query_cache_size, ttl_s=settings.query_cache_ttl_s)

    # ------------------------------------------------------------------ filters
    def resolve_document_filter(self, s: Session, f: QueryFilters | None) -> list[str] | None:
        """None => no filter (all ready docs). [] => filter matched nothing."""
        if f is None or f.is_empty():
            return None
        stmt = select(Document.id).where(Document.status == DocumentStatus.READY)
        if f.document_ids:
            stmt = stmt.where(Document.id.in_(f.document_ids))
        if f.file_types:
            stmt = stmt.where(Document.file_type.in_(f.file_types))
        if f.languages:
            stmt = stmt.where(Document.language.in_([l.lower() for l in f.languages]))
        if f.uploaded_by:
            stmt = stmt.where(Document.uploaded_by == f.uploaded_by)
        if f.filename_contains:
            stmt = stmt.where(Document.filename.ilike(f"%{f.filename_contains}%"))
        if f.created_after:
            stmt = stmt.where(Document.created_at >= f.created_after)
        if f.created_before:
            stmt = stmt.where(Document.created_at < f.created_before)
        if f.tags:
            stmt = stmt.where(
                Document.id.in_(select(DocumentTag.document_id).where(DocumentTag.tag.in_(f.tags)))
            )
        for key, value in (f.metadata or {}).items():
            stmt = stmt.where(Document.extra_metadata[key].as_string() == str(value))
        return list(s.execute(stmt).scalars())

    # ------------------------------------------------------------------- search
    def _query_vector(self, query: str) -> list[float]:
        key = (self.embedder.model_name, query)
        vec = self._embedding_cache.get(key)
        if vec is None:
            vec = self.embedder.embed_query(query)
            self._embedding_cache.set(key, vec)
        return vec

    def search(
        self,
        query: str,
        top_k: int,
        filters: QueryFilters | None,
        rerank: bool | None = None,
        hybrid: bool | None = None,
        min_score: float | None = None,
    ) -> SearchOutcome:
        use_rerank = (self.settings.reranker_enabled if rerank is None else rerank) and self.reranker is not None
        use_hybrid = (self.settings.hybrid_search if hybrid is None else hybrid) and self.keyword.enabled
        cache_key = (
            " ".join(query.lower().split()),
            json.dumps(filters.model_dump(mode="json") if filters else None, sort_keys=True),
            top_k, use_rerank, use_hybrid, min_score, self.corpus_version.value,
        )
        cached = self._result_cache.get(cache_key)
        if cached is not None:
            return SearchOutcome(cached.candidates, cached.reranked, cached.hybrid, cache_hit=True)

        timings: dict[str, float] = {}
        t0 = time.perf_counter()
        n = max(self.settings.retrieval_candidates, top_k * 3)
        with self.sessions() as s:
            allowed = self.resolve_document_filter(s, filters)
            if allowed is not None and not allowed:
                return SearchOutcome([], use_rerank, use_hybrid, timings_ms={"total": 0.0})

            qvec = self._query_vector(query)
            timings["embed"] = (time.perf_counter() - t0) * 1000

            t1 = time.perf_counter()
            vec_hits = self.vectors.query(qvec, n, allowed)
            timings["vector_search"] = (time.perf_counter() - t1) * 1000

            kw_hits = []
            if use_hybrid:
                t1 = time.perf_counter()
                kw_hits = self.keyword.search(s, query, n, allowed)
                timings["keyword_search"] = (time.perf_counter() - t1) * 1000

            # Reciprocal Rank Fusion: score = sum 1/(k + rank). Rank-based, so the two
            # retrievers' incomparable score scales never need normalizing.
            cands: dict[str, Candidate] = {}
            for rank, h in enumerate(vec_hits, 1):
                c = cands.setdefault(h.chunk_id, Candidate(h.chunk_id))
                c.vector_similarity, c.vector_rank = h.similarity, rank
                c.rrf += 1.0 / (self.settings.rrf_k + rank)
            for rank, h in enumerate(kw_hits, 1):
                c = cands.setdefault(h.chunk_id, Candidate(h.chunk_id))
                c.keyword_rank = rank
                c.rrf += 1.0 / (self.settings.rrf_k + rank)
            ranked = sorted(cands.values(), key=lambda c: c.rrf, reverse=True)

            # Hydrate from the source of truth; drop anything not READY (soft-deleted docs,
            # in-flight re-ingestion, orphaned vectors from a partially failed purge).
            if ranked:
                rows = s.execute(
                    select(Chunk, Document)
                    .join(Document, Document.id == Chunk.document_id)
                    .where(Chunk.id.in_([c.chunk_id for c in ranked]))
                    .where(Document.status == DocumentStatus.READY)
                ).all()
                by_id = {chunk.id: (chunk, doc) for chunk, doc in rows}
                ranked = [c for c in ranked if c.chunk_id in by_id]
                for c in ranked:
                    c.chunk, c.document = by_id[c.chunk_id]

        # The cross-encoder re-scores the head of the fused list.
        if use_rerank and ranked:
            t1 = time.perf_counter()
            head = ranked[: self.settings.rerank_candidates]
            scores = self.reranker.score(query, [rerank_text(c) for c in head])
            for c, sc in zip(head, scores):
                c.rerank = sc
            ranked = sorted(head, key=lambda c: c.rerank, reverse=True) + ranked[len(head):]
            timings["rerank"] = (time.perf_counter() - t1) * 1000

        if min_score is not None:
            ranked = [c for c in ranked if final_score(c, use_rerank, use_hybrid) >= min_score]
        result = ranked[:top_k]
        timings["total"] = (time.perf_counter() - t0) * 1000
        outcome = SearchOutcome(result, use_rerank, use_hybrid, timings_ms={k: round(v, 1) for k, v in timings.items()})
        self._result_cache.set(cache_key, outcome)
        return outcome


def rerank_text(c: Candidate) -> str:
    """Text the cross-encoder reads. For code, prefixing file and symbol name gives the
    web-trained reranker context it cannot infer from a bare method body (measured: it
    moved `report_failure` from rank 3 to 1 for "what happens when a proxy fails?")."""
    if c.chunk.chunk_type == "code" and c.chunk.section:
        return f"{c.document.filename} | {c.chunk.section}\n{c.chunk.content}"
    return c.chunk.content


def final_score(c: Candidate, reranked: bool, hybrid: bool) -> float:
    """The score results are ordered by: reranker relevance (0-1) when reranking, the RRF
    score for un-reranked hybrid search, otherwise vector cosine similarity."""
    if reranked and c.rerank is not None:
        return c.rerank
    if hybrid or c.vector_similarity is None:
        return c.rrf
    return c.vector_similarity

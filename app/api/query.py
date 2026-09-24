import logging
import re
import time
import uuid

from fastapi import APIRouter, Depends

from app.api.deps import get_container, get_current_user
from app.container import Container
from app.db.models import QueryLog
from app.errors import AppError
from app.gateway import RAG_SYSTEM_PROMPT, build_rag_messages
from app.retrieval.search import final_score
from app.schemas import Answer, QueryRequest, QueryResponse, QueryResult, ScoreBreakdown

log = logging.getLogger(__name__)
router = APIRouter(prefix="/v1", tags=["query"])


@router.post("/query", response_model=QueryResponse, summary="Semantic search over the knowledge base")
def query(req: QueryRequest, user: str = Depends(get_current_user), c: Container = Depends(get_container)):
    started = time.perf_counter()
    outcome = c.search.search(
        req.query, req.top_k, req.filters, rerank=req.rerank, hybrid=req.hybrid, min_score=req.min_score
    )

    results = []
    for rank, cand in enumerate(outcome.candidates, 1):
        ch, doc = cand.chunk, cand.document
        results.append(
            QueryResult(
                rank=rank,
                score=round(final_score(cand, outcome.reranked, outcome.hybrid), 4),
                chunk_id=ch.id,
                document_id=doc.id,
                filename=doc.filename,
                file_type=doc.file_type,
                language=doc.language,
                chunk_index=ch.chunk_index,
                section=ch.section,
                page_start=ch.page_start,
                page_end=ch.page_end,
                start_line=ch.start_line,
                end_line=ch.end_line,
                content=ch.content,
                scores=ScoreBreakdown(
                    vector_similarity=None if cand.vector_similarity is None else round(cand.vector_similarity, 4),
                    vector_rank=cand.vector_rank,
                    keyword_rank=cand.keyword_rank,
                    rrf=round(cand.rrf, 5),
                    rerank=None if cand.rerank is None else round(cand.rerank, 4),
                ),
            )
        )

    answer = None
    if req.generate_answer:
        answer = _generate_answer(c, user, req.query, results)

    timings = dict(outcome.timings_ms)
    timings["request_total"] = round((time.perf_counter() - started) * 1000, 1)
    query_id = str(uuid.uuid4())
    _log_query(c, query_id, user, req, results, timings["request_total"], outcome, answer)
    return QueryResponse(
        query_id=query_id,
        query=req.query,
        results=results,
        answer=answer,
        reranked=outcome.reranked,
        hybrid=outcome.hybrid,
        cache_hit=outcome.cache_hit,
        timings_ms=timings,
    )


def _generate_answer(c: Container, user: str, question: str, results: list[QueryResult]) -> Answer:
    if not results:
        return Answer(text="No relevant content was found in the knowledge base.", model=None)
    try:
        llm = c.gateway.complete(
            user_id=user,
            purpose="rag_answer",
            system=RAG_SYSTEM_PROMPT,
            messages=build_rag_messages(question, [r.model_dump() for r in results]),
        )
    except AppError as exc:
        # Retrieval results are still useful - degrade gracefully instead of failing the query.
        log.warning("Answer generation failed: %s", exc.message)
        return Answer(text=None, model=None, error=exc.message)
    if llm.stop_reason == "refusal":
        return Answer(text=None, model=llm.model, error="The model declined to answer this request")
    cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", llm.text) if 1 <= int(n) <= len(results)})
    return Answer(text=llm.text, model=llm.model, citations=cited)


def _log_query(c, query_id, user, req, results, latency_ms, outcome, answer) -> None:
    try:
        with c.sessions.begin() as s:
            s.add(
                QueryLog(
                    query_id=query_id,
                    user_id=user,
                    query_text=req.query,
                    filters=req.filters.model_dump(mode="json", exclude_none=True) if req.filters else None,
                    top_k=req.top_k,
                    result_count=len(results),
                    result_chunk_ids=[r.chunk_id for r in results],
                    top_score=results[0].score if results else None,
                    latency_ms=latency_ms,
                    cache_hit=outcome.cache_hit,
                    reranked=outcome.reranked,
                    answer_generated=bool(answer and answer.text),
                )
            )
    except Exception:  # analytics must never fail the user's query
        log.exception("Failed to write query log")

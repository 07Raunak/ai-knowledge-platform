from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from sqlalchemy import func, select, text

from app.api.deps import get_container
from app.container import Container
from app.db.models import Document, DocumentStatus

router = APIRouter(tags=["health"])


@router.get("/health", summary="Liveness probe")
def health():
    return {"status": "ok"}


@router.get("/ready", summary="Readiness probe with dependency checks and queue depth")
def ready(c: Container = Depends(get_container)):
    checks: dict = {}
    ok = True
    try:
        with c.sessions() as s:
            s.execute(text("SELECT 1"))
            counts = dict(s.execute(select(Document.status, func.count()).group_by(Document.status)).all())
            stuck_purges = s.execute(
                select(func.count()).where(
                    Document.status == DocumentStatus.DELETED,
                    Document.purge_attempts >= c.settings.max_purge_attempts,
                )
            ).scalar_one()
        checks["database"] = "ok"
        checks["documents_by_status"] = counts
        checks["purges_needing_attention"] = stuck_purges
    except Exception as exc:
        ok = False
        checks["database"] = f"error: {exc}"
    try:
        checks["vector_store"] = {"status": "ok", "vectors": c.vectors.count(), "collection": c.vectors.collection_name}
    except Exception as exc:
        ok = False
        checks["vector_store"] = f"error: {exc}"
    checks["embedding_model"] = c.embedder.model_name
    checks["reranker"] = c.settings.reranker_model if c.reranker else "disabled"
    checks["llm_gateway"] = c.settings.llm_model if c.gateway.enabled else "disabled"
    return JSONResponse(status_code=200 if ok else 503, content={"status": "ok" if ok else "degraded", "checks": checks})

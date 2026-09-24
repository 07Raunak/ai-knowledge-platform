import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

from app.api import documents, health, llm, query
from app.config import Settings
from app.container import build_container
from app.errors import register_exception_handlers
from app.logging_config import configure_logging, request_id_var

log = logging.getLogger("app")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    configure_logging(settings.log_level, settings.log_json)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        container = build_container(settings)
        app.state.container = container
        if settings.environment != "test":
            # Load models at startup so the first user request doesn't pay for it.
            container.embedder.embed_query("warmup")
            if container.reranker:
                container.reranker.warmup()
        container.worker.start()
        log.info("%s started (db=%s)", settings.app_name, settings.resolved_database_url)
        yield
        container.worker.stop()
        container.engine.dispose()

    app = FastAPI(
        title=settings.app_name,
        version="1.0.0",
        description=(
            "Upload documents and code, search them semantically (hybrid vector + keyword "
            "retrieval with cross-encoder reranking), and access LLMs through a centralized gateway."
        ),
        lifespan=lifespan,
    )
    register_exception_handlers(app)

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:16]
        token = request_id_var.set(rid)
        started = time.perf_counter()
        try:
            response = await call_next(request)
        finally:
            request_id_var.reset(token)
        elapsed = (time.perf_counter() - started) * 1000
        response.headers["X-Request-ID"] = rid
        log.info("%s %s -> %s (%.0f ms) [%s]", request.method, request.url.path, response.status_code, elapsed, rid)
        return response

    app.include_router(health.router)
    app.include_router(documents.router)
    app.include_router(query.router)
    app.include_router(llm.router)
    return app


app = create_app()

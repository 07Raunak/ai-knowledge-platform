"""Composition root: builds every long-lived dependency once per application."""

from dataclasses import dataclass

from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import Settings
from app.db.session import create_db_engine, init_db, make_session_factory
from app.embeddings import Embedder, build_embedder
from app.gateway import LLMGateway
from app.ingestion.pipeline import IngestionPipeline
from app.retrieval.cache import CorpusVersion
from app.retrieval.keyword import KeywordIndex
from app.retrieval.reranker import CrossEncoderReranker
from app.retrieval.search import SearchService
from app.storage import LocalFileStorage
from app.vectorstore import ChromaVectorStore, VectorStore
from app.worker import Worker


@dataclass
class Container:
    settings: Settings
    engine: Engine
    sessions: sessionmaker[Session]
    storage: LocalFileStorage
    embedder: Embedder
    vectors: VectorStore
    keyword: KeywordIndex
    reranker: CrossEncoderReranker | None
    corpus_version: CorpusVersion
    pipeline: IngestionPipeline
    search: SearchService
    gateway: LLMGateway
    worker: Worker


def build_container(settings: Settings) -> Container:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    engine = create_db_engine(settings.resolved_database_url)
    init_db(engine)
    sessions = make_session_factory(engine)
    storage = LocalFileStorage(settings.upload_dir)
    embedder = build_embedder(settings)
    vectors = ChromaVectorStore(settings.chroma_dir, embedder.model_name)
    keyword = KeywordIndex(enabled=engine.dialect.name == "sqlite")
    reranker = CrossEncoderReranker(settings.reranker_model) if settings.reranker_enabled else None
    corpus_version = CorpusVersion()
    pipeline = IngestionPipeline(settings, sessions, storage, embedder, vectors, keyword, corpus_version)
    search = SearchService(settings, sessions, embedder, vectors, keyword, reranker, corpus_version)
    gateway = LLMGateway(settings, sessions)
    worker = Worker(settings, sessions, pipeline)
    return Container(settings, engine, sessions, storage, embedder, vectors, keyword, reranker,
                     corpus_version, pipeline, search, gateway, worker)

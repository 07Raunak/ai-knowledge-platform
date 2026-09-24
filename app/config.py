"""Application settings, loaded from environment variables (prefix ``KP_``) or a ``.env`` file."""

from functools import cached_property
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="KP_", env_file=".env", extra="ignore")

    app_name: str = "Internal AI Knowledge Platform"
    environment: Literal["dev", "test", "prod"] = "dev"
    log_level: str = "INFO"
    log_json: bool = False

    # --- Storage -----------------------------------------------------------
    data_dir: Path = Path("data")
    database_url: str | None = None  # defaults to SQLite inside data_dir
    max_upload_mb: int = 25

    # --- Auth --------------------------------------------------------------
    # Comma-separated "api_key:user_id" pairs. Empty => auth disabled (local dev only).
    api_keys: str = ""

    # --- Embeddings --------------------------------------------------------
    embedding_backend: Literal["sentence-transformers", "hash"] = "sentence-transformers"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    # BGE models are trained with an instruction prefix on the *query* side only.
    embedding_query_prefix: str = "Represent this sentence for searching relevant passages: "
    embedding_batch_size: int = 32

    # --- Extraction --------------------------------------------------------
    ocr_enabled: bool = True  # OCR PDF pages that have no text layer
    ocr_dpi: int = 150
    ocr_rescue_dpi: int = 300  # re-read low-confidence lines at this resolution

    # --- Chunking ----------------------------------------------------------
    chunk_size_tokens: int = 350  # bge-small max sequence length is 512 tokens
    chunk_overlap_tokens: int = 50

    # --- Retrieval ---------------------------------------------------------
    hybrid_search: bool = True  # vector + BM25 keyword search fused with RRF
    retrieval_candidates: int = 30  # candidates pulled from each retriever
    rrf_k: int = 60
    reranker_enabled: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_candidates: int = 12  # cross-encoder cost is linear in this (~80 ms/pair on a laptop CPU)
    query_cache_ttl_s: int = 300
    query_cache_size: int = 1024

    # --- Background worker -------------------------------------------------
    worker_threads: int = 2
    worker_poll_interval_s: float = 2.0
    worker_lease_s: int = 600  # a crashed worker's job becomes claimable after this
    max_ingest_attempts: int = 3
    max_purge_attempts: int = 10
    retry_base_delay_s: float = 5.0

    # --- LLM gateway -------------------------------------------------------
    llm_provider: Literal["anthropic", "disabled"] = "anthropic"
    llm_model: str = "claude-opus-5"
    llm_allowed_models: list[str] = Field(default_factory=lambda: ["claude-opus-5"])
    llm_effort: Literal["low", "medium", "high", "xhigh", "max"] = "medium"
    llm_max_tokens: int = 4096
    llm_timeout_s: float = 120.0
    llm_max_retries: int = 3
    llm_refusal_fallbacks: bool = True
    llm_rate_limit_per_minute: int = 20

    @cached_property
    def resolved_database_url(self) -> str:
        return self.database_url or f"sqlite:///{(self.data_dir / 'knowledge.db').as_posix()}"

    @property
    def upload_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"

    @cached_property
    def api_key_map(self) -> dict[str, str]:
        pairs = [p.strip() for p in self.api_keys.split(",") if p.strip()]
        result: dict[str, str] = {}
        for pair in pairs:
            key, _, user = pair.partition(":")
            if key and user:
                result[key] = user
        return result

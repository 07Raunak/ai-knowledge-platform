"""Embedding providers behind one small interface so the model can be swapped
(local sentence-transformers today, a hosted embedding API via the gateway tomorrow)."""

import hashlib
import logging
import math
import re
import threading
from typing import Protocol

log = logging.getLogger(__name__)


class Embedder(Protocol):
    model_name: str
    dimension: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str, query_prefix: str = "", batch_size: int = 32):
        self.model_name = model_name
        self.query_prefix = query_prefix
        self.batch_size = batch_size
        self._model = None
        self._lock = threading.Lock()

    def _load(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from sentence_transformers import SentenceTransformer

                    log.info("Loading embedding model %s", self.model_name)
                    self._model = SentenceTransformer(self.model_name, device="cpu")
        return self._model

    @property
    def dimension(self) -> int:
        model = self._load()
        getter = getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension
        return int(getter())

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self._load().encode(
            texts, batch_size=self.batch_size, normalize_embeddings=True, show_progress_bar=False
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        vector = self._load().encode(
            [self.query_prefix + text], normalize_embeddings=True, show_progress_bar=False
        )
        return vector[0].tolist()


class HashingEmbedder:
    """Deterministic bag-of-words feature hashing. No model download - used by tests and
    as a degraded-mode fallback. Lexical only, so retrieval quality is far lower."""

    model_name = "hashing-bow-256"
    dimension = 256

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dimension
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            h = int(hashlib.md5(token.encode()).hexdigest(), 16)
            vec[h % self.dimension] += 1.0 if (h >> 8) & 1 else -1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def build_embedder(settings) -> Embedder:
    if settings.embedding_backend == "hash":
        return HashingEmbedder()
    return SentenceTransformerEmbedder(
        settings.embedding_model, settings.embedding_query_prefix, settings.embedding_batch_size
    )

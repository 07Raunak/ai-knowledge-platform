"""Cross-encoder reranking. The bi-encoder retrieves candidates cheaply; the cross-encoder
reads (query, chunk) pairs jointly and is much more precise, so it re-orders the top-N."""

import logging
import math
import threading

log = logging.getLogger(__name__)


class CrossEncoderReranker:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None
        self._lock = threading.Lock()

    def _load(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from sentence_transformers import CrossEncoder

                    log.info("Loading reranker %s", self.model_name)
                    self._model = CrossEncoder(self.model_name, device="cpu")
        return self._model

    def warmup(self) -> None:
        self._load()

    def score(self, query: str, passages: list[str]) -> list[float]:
        """Returns relevance probabilities in (0, 1) (sigmoid of the cross-encoder logit)."""
        if not passages:
            return []
        logits = self._load().predict([(query, p) for p in passages], show_progress_bar=False)
        return [1.0 / (1.0 + math.exp(-float(l))) for l in logits]

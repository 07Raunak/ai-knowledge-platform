"""Centralized AI Gateway.

Every LLM call in the platform (the RAG answer step and the developer-facing chat endpoint)
goes through here, which gives one place for:
  * provider credentials (developers never hold provider API keys)
  * model allow-listing
  * per-user rate limiting
  * retries / timeouts / refusal fallbacks
  * usage + cost attribution logging (``llm_usage`` table)
"""

import logging
import threading
import time
from dataclasses import dataclass, field

from sqlalchemy.orm import Session, sessionmaker

from app.config import Settings
from app.db.models import LLMUsage
from app.errors import AppError, RateLimitedError, ServiceUnavailableError, UpstreamError

log = logging.getLogger(__name__)


@dataclass
class LLMResult:
    text: str
    model: str
    stop_reason: str | None
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    fallback_used: bool = False


@dataclass
class _Bucket:
    tokens: float
    updated: float = field(default_factory=time.monotonic)


class RateLimiter:
    """Per-user token bucket (in-process). With several replicas this becomes a Redis
    ``INCR``/``EXPIRE`` or GCRA script keyed by user."""

    def __init__(self, per_minute: int):
        self.capacity = float(per_minute)
        self.refill_per_s = per_minute / 60.0
        self._buckets: dict[str, _Bucket] = {}
        self._lock = threading.Lock()

    def acquire(self, user_id: str) -> None:
        with self._lock:
            now = time.monotonic()
            b = self._buckets.setdefault(user_id, _Bucket(self.capacity, now))
            b.tokens = min(self.capacity, b.tokens + (now - b.updated) * self.refill_per_s)
            b.updated = now
            if b.tokens < 1:
                retry_after = (1 - b.tokens) / self.refill_per_s
                raise RateLimitedError(
                    "LLM rate limit exceeded", details={"retry_after_s": round(retry_after, 1)}
                )
            b.tokens -= 1


class LLMGateway:
    def __init__(self, settings: Settings, session_factory: sessionmaker[Session]):
        self.settings = settings
        self._sessions = session_factory
        self._limiter = RateLimiter(settings.llm_rate_limit_per_minute)
        self._client = None

    @property
    def enabled(self) -> bool:
        return self.settings.llm_provider != "disabled"

    def _get_client(self):
        if self._client is None:
            import anthropic

            try:
                # Credentials resolve from the environment (ANTHROPIC_API_KEY, or an
                # `ant auth login` profile). The SDK retries 408/409/429/5xx with backoff.
                self._client = anthropic.Anthropic(
                    timeout=self.settings.llm_timeout_s, max_retries=self.settings.llm_max_retries
                )
            except anthropic.AnthropicError as exc:
                raise ServiceUnavailableError(f"LLM provider not configured: {exc}") from exc
        return self._client

    def complete(
        self,
        *,
        user_id: str,
        purpose: str,
        messages: list[dict],
        system: str | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
    ) -> LLMResult:
        if not self.enabled:
            raise ServiceUnavailableError("LLM gateway is disabled (KP_LLM_PROVIDER=disabled)")
        model = model or self.settings.llm_model
        if model not in self.settings.llm_allowed_models:
            raise AppError(
                f"Model '{model}' is not allowed", details={"allowed": self.settings.llm_allowed_models}
            )
        self._limiter.acquire(user_id)

        import anthropic

        params: dict = dict(
            model=model,
            max_tokens=max_tokens or self.settings.llm_max_tokens,
            messages=messages,
            output_config={"effort": self.settings.llm_effort},
        )
        if system:
            params["system"] = system
        if self.settings.llm_refusal_fallbacks:
            # On a safety decline, the API re-runs the request on Anthropic's recommended
            # fallback model inside the same call instead of returning the refusal.
            params["betas"] = ["server-side-fallback-2026-07-01"]
            params["fallbacks"] = "default"

        started = time.perf_counter()
        status, error = "error", None
        try:
            response = self._get_client().beta.messages.create(**params)
            text = "".join(b.text for b in response.content if b.type == "text")
            fallback_used = any(
                getattr(e, "type", None) == "fallback_message"
                for e in (getattr(response.usage, "iterations", None) or [])
            )
            status = "refused" if response.stop_reason == "refusal" else "ok"
            result = LLMResult(
                text=text,
                model=response.model,
                stop_reason=response.stop_reason,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                latency_ms=(time.perf_counter() - started) * 1000,
                fallback_used=fallback_used,
            )
            self._log_usage(user_id, purpose, response.model, status, result.input_tokens,
                            result.output_tokens, result.latency_ms, None)
            return result
        except anthropic.RateLimitError as exc:
            error = "provider rate limit"
            raise RateLimitedError("Upstream LLM provider is rate limiting; retry later") from exc
        except anthropic.AuthenticationError as exc:
            error = "provider authentication failed"
            raise ServiceUnavailableError("LLM provider credentials are invalid or missing") from exc
        except anthropic.BadRequestError as exc:
            error = exc.message
            raise AppError(f"LLM request rejected: {exc.message}") from exc
        except anthropic.APIStatusError as exc:
            error = f"HTTP {exc.status_code}: {exc.message}"
            raise UpstreamError(f"LLM provider error ({exc.status_code})") from exc
        except anthropic.APIConnectionError as exc:
            error = "connection error"
            raise UpstreamError("Could not reach the LLM provider") from exc
        except anthropic.AnthropicError as exc:  # e.g. no credentials found
            error = str(exc)
            raise ServiceUnavailableError(f"LLM provider not configured: {exc}") from exc
        finally:
            if error is not None:
                self._log_usage(user_id, purpose, model, status, 0, 0,
                                (time.perf_counter() - started) * 1000, error)

    def _log_usage(self, user_id, purpose, model, status, tin, tout, latency_ms, error) -> None:
        try:
            with self._sessions.begin() as s:
                s.add(LLMUsage(user_id=user_id, purpose=purpose, model=model, status=status,
                               input_tokens=tin, output_tokens=tout, latency_ms=latency_ms, error=error))
        except Exception:  # usage logging must never break the request path
            log.exception("Failed to record LLM usage")


RAG_SYSTEM_PROMPT = """You answer questions for internal developers using ONLY the numbered context \
passages provided from the company knowledge base (documents and source code).

- Cite the passages you rely on inline as [1], [2], etc.
- If the context does not contain the answer, say so plainly instead of guessing.
- For code questions, reference function/method names and line numbers when they are given.
- Be concise."""


def build_rag_messages(question: str, passages: list[dict]) -> list[dict]:
    blocks = []
    for i, p in enumerate(passages, 1):
        where = p["filename"]
        if p.get("start_line"):
            where += f" L{p['start_line']}-{p['end_line']}"
        elif p.get("page_start"):
            where += f" p.{p['page_start']}"
        if p.get("section"):
            where += f" ({p['section']})"
        blocks.append(f"[{i}] {where}\n{p['content']}")
    context = "\n\n---\n\n".join(blocks)
    return [
        {
            "role": "user",
            "content": f"<context>\n{context}\n</context>\n\nQuestion: {question}",
        }
    ]

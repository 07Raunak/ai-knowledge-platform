"""Durable, DB-backed job processing for ingestion and purges.

The ``documents`` table *is* the queue: a job is claimed with an atomic compare-and-set
UPDATE that also sets a lease (``locked_until``). If a worker crashes mid-job the lease
expires and another worker re-claims it - no job is lost across restarts, and several
worker processes can safely share one database. Swapping in Celery/SQS later only changes
how jobs are *signalled*, not the state machine.

    queued --claim--> processing --ok--> ready
       ^                   | error (retryable, attempts < max)
       +---- backoff ------+
                           | error (permanent or attempts exhausted) --> failed
    any --DELETE--> deleted --purge ok--> (row removed)
                        ^--- purge error: backoff + retry ---+
"""

import logging
import threading
from datetime import datetime, timedelta, timezone

from sqlalchemy import and_, or_, select, update
from sqlalchemy.orm import Session, sessionmaker

from app.config import Settings
from app.db.models import Document, DocumentStatus
from app.errors import ExtractionError
from app.ingestion.pipeline import IngestionPipeline

log = logging.getLogger(__name__)


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Worker:
    def __init__(self, settings: Settings, sessions: sessionmaker[Session], pipeline: IngestionPipeline):
        self.settings = settings
        self.sessions = sessions
        self.pipeline = pipeline
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []

    # ---------------------------------------------------------------- lifecycle
    def start(self) -> None:
        for i in range(self.settings.worker_threads):
            t = threading.Thread(target=self._run, name=f"ingest-worker-{i}", daemon=True)
            t.start()
            self._threads.append(t)
        log.info("Started %d worker thread(s)", len(self._threads))

    def stop(self, timeout: float = 10.0) -> None:
        self._stop.set()
        self._wake.set()
        for t in self._threads:
            t.join(timeout)

    def notify(self) -> None:
        """Wake an idle worker immediately (called after upload / delete)."""
        self._wake.set()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                did_work = self.run_once()
            except Exception:
                log.exception("Worker loop error")
                did_work = False
            if not did_work:
                self._wake.wait(self.settings.worker_poll_interval_s)
                self._wake.clear()

    def run_once(self) -> bool:
        """Process at most one purge and one ingestion job. Returns True if work was done."""
        purged = self._run_purge()
        ingested = self._run_ingest()
        return purged or ingested

    # ------------------------------------------------------------------ claiming
    def _claim(self, statuses_filter, new_status: str) -> str | None:
        now = _now()
        lease = now + timedelta(seconds=self.settings.worker_lease_s)
        with self.sessions.begin() as s:
            candidate = s.execute(
                select(Document.id)
                .where(statuses_filter(now))
                .order_by(Document.created_at)
                .limit(1)
            ).scalar_one_or_none()
            if candidate is None:
                return None
            # Compare-and-set: only one worker wins the row.
            res = s.execute(
                update(Document)
                .where(Document.id == candidate, statuses_filter(now))
                .values(status=new_status, locked_until=lease)
            )
            return candidate if res.rowcount == 1 else None

    def _run_ingest(self) -> bool:
        def claimable(now):
            return or_(
                and_(
                    Document.status == DocumentStatus.QUEUED,
                    or_(Document.next_attempt_at.is_(None), Document.next_attempt_at <= now),
                ),
                # lease expired => the previous worker died mid-job
                and_(Document.status == DocumentStatus.PROCESSING, Document.locked_until < now),
            )

        doc_id = self._claim(claimable, DocumentStatus.PROCESSING)
        if doc_id is None:
            return False
        try:
            self.pipeline.ingest(doc_id)
        except ExtractionError as exc:
            self._fail(doc_id, str(exc), permanent=True)
        except Exception as exc:
            log.exception("Ingestion failed for %s", doc_id)
            self._fail(doc_id, f"{type(exc).__name__}: {exc}", permanent=False)
        return True

    def _fail(self, doc_id: str, error: str, permanent: bool) -> None:
        with self.sessions.begin() as s:
            doc = s.get(Document, doc_id)
            if doc is None or doc.status != DocumentStatus.PROCESSING:
                return
            doc.attempts += 1
            doc.error = error
            doc.locked_until = None
            if permanent or doc.attempts >= self.settings.max_ingest_attempts:
                doc.status = DocumentStatus.FAILED
                log.warning("Document %s failed permanently: %s", doc_id, error)
            else:
                delay = self.settings.retry_base_delay_s * (2 ** (doc.attempts - 1))
                doc.status = DocumentStatus.QUEUED
                doc.next_attempt_at = _now() + timedelta(seconds=delay)
                log.warning("Document %s attempt %d failed, retrying in %.0fs", doc_id, doc.attempts, delay)

    # -------------------------------------------------------------------- purge
    def _purge_due(self, now: datetime, *, ignore_backoff: bool = False):
        conds = [
            Document.status == DocumentStatus.DELETED,
            or_(Document.locked_until.is_(None), Document.locked_until < now),
        ]
        if not ignore_backoff:
            conds += [
                Document.purge_attempts < self.settings.max_purge_attempts,
                or_(Document.next_attempt_at.is_(None), Document.next_attempt_at <= now),
            ]
        return and_(*conds)

    def _claim_purge(self, doc_id: str, *, ignore_backoff: bool = False) -> bool:
        """Atomically take the purge lease so that exactly one purger (a background worker
        or an inline ``?hard=true`` request) works on a document at a time."""
        now = _now()
        with self.sessions.begin() as s:
            res = s.execute(
                update(Document)
                .where(Document.id == doc_id, self._purge_due(now, ignore_backoff=ignore_backoff))
                .values(locked_until=now + timedelta(seconds=self.settings.worker_lease_s))
            )
            return res.rowcount == 1

    def _run_purge(self) -> bool:
        with self.sessions() as s:
            doc_id = s.execute(
                select(Document.id).where(self._purge_due(_now())).limit(1)
            ).scalar_one_or_none()
        if doc_id is None or not self._claim_purge(doc_id):
            return False
        self._purge_claimed(doc_id)
        return True

    def purge_now(self, doc_id: str) -> bool:
        """Synchronous purge for ``DELETE ?hard=true``. Returns False if another purger holds
        the lease or the purge failed (it is then retried in the background)."""
        if not self._claim_purge(doc_id, ignore_backoff=True):
            return False
        return self._purge_claimed(doc_id)

    def _purge_claimed(self, doc_id: str) -> bool:
        try:
            self.pipeline.purge(doc_id)
            return True
        except Exception as exc:
            log.exception("Purge failed for %s", doc_id)
            with self.sessions.begin() as s:
                doc = s.get(Document, doc_id)
                if doc is not None:
                    doc.purge_attempts += 1
                    doc.error = f"purge failed: {type(exc).__name__}: {exc}"
                    doc.locked_until = None
                    delay = self.settings.retry_base_delay_s * (2 ** min(doc.purge_attempts, 8))
                    doc.next_attempt_at = _now() + timedelta(seconds=delay)
                    if doc.purge_attempts >= self.settings.max_purge_attempts:
                        # Still invisible to users; surfaces on /ready for an operator.
                        log.error("Document %s purge exhausted retries", doc_id)
            return False

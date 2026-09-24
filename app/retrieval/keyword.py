"""BM25 keyword retrieval over SQLite FTS5.

Dense embeddings are weak at exact identifiers (``report_failure``, error codes, config
keys) - exactly what developers search for in code. A lexical retriever covers that gap.
"""

import re
from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.orm import Session

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does", "for", "from", "how",
    "i", "if", "in", "is", "it", "of", "on", "or", "that", "the", "this", "to", "was", "what",
    "when", "where", "which", "who", "why", "with", "you", "your", "me", "my", "we", "our",
}


@dataclass
class KeywordHit:
    chunk_id: str
    bm25: float  # FTS5 bm25(): lower (more negative) is better


def build_match_query(query: str) -> str | None:
    terms = [t for t in re.findall(r"[A-Za-z0-9]+", query.lower()) if t not in _STOPWORDS and len(t) > 1]
    if not terms:
        return None
    # Quote every term: neutralizes FTS5 syntax (AND/OR/NEAR/*, column filters) in user input.
    return " OR ".join(f'"{t}"' for t in dict.fromkeys(terms))


class KeywordIndex:
    def __init__(self, enabled: bool):
        self.enabled = enabled

    def add(self, session: Session, rows: list[tuple[str, str, str]]) -> None:
        """rows: (chunk_id, document_id, content)"""
        if not self.enabled or not rows:
            return
        session.execute(
            text("INSERT INTO chunks_fts (content, chunk_id, document_id) VALUES (:c, :cid, :did)"),
            [{"c": c, "cid": cid, "did": did} for cid, did, c in rows],
        )

    def delete_document(self, session: Session, document_id: str) -> None:
        if self.enabled:
            session.execute(text("DELETE FROM chunks_fts WHERE document_id = :d"), {"d": document_id})

    def search(
        self, session: Session, query: str, n: int, document_ids: list[str] | None = None
    ) -> list[KeywordHit]:
        if not self.enabled:
            return []
        match = build_match_query(query)
        if match is None or (document_ids is not None and not document_ids):
            return []
        sql = "SELECT chunk_id, bm25(chunks_fts) AS score FROM chunks_fts WHERE chunks_fts MATCH :m"
        params: dict = {"m": match, "n": n}
        if document_ids:
            keys = [f"d{i}" for i in range(len(document_ids))]
            sql += f" AND document_id IN ({', '.join(':' + k for k in keys)})"
            params.update(dict(zip(keys, document_ids)))
        sql += " ORDER BY score LIMIT :n"
        return [KeywordHit(r.chunk_id, r.score) for r in session.execute(text(sql), params)]

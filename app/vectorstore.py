"""Vector store abstraction. Chroma (embedded, HNSW, cosine) is the local implementation;
docs/database_schema.md shows the equivalent pgvector table for production."""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass
class VectorHit:
    chunk_id: str
    similarity: float  # cosine similarity in [-1, 1]


class VectorStore(Protocol):
    collection_name: str

    def upsert(self, ids: list[str], vectors: list[list[float]], metadatas: list[dict]) -> None: ...

    def query(self, vector: list[float], n: int, document_ids: list[str] | None = None) -> list[VectorHit]: ...

    def delete_document(self, document_id: str) -> None: ...

    def count(self) -> int: ...


def collection_name_for(model_name: str) -> str:
    """One collection per embedding model: vectors from different models are never mixed,
    and a model upgrade is a re-embed into a new collection + switch-over."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", model_name).strip("_").lower()
    return f"chunks_{slug}"[:60]


class ChromaVectorStore:
    def __init__(self, persist_dir: Path, model_name: str):
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(persist_dir), settings=ChromaSettings(anonymized_telemetry=False)
        )
        self.collection_name = collection_name_for(model_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine", "embedding_model": model_name},
            embedding_function=None,
        )

    def upsert(self, ids, vectors, metadatas) -> None:
        batch = 500
        for i in range(0, len(ids), batch):
            self._collection.upsert(
                ids=ids[i : i + batch], embeddings=vectors[i : i + batch], metadatas=metadatas[i : i + batch]
            )

    def query(self, vector, n, document_ids=None) -> list[VectorHit]:
        if document_ids is not None and not document_ids:
            return []
        where = None
        if document_ids:
            where = (
                {"document_id": document_ids[0]}
                if len(document_ids) == 1
                else {"document_id": {"$in": list(document_ids)}}
            )
        total = self._collection.count()
        if total == 0:
            return []
        res = self._collection.query(
            query_embeddings=[vector], n_results=min(n, total), where=where, include=["distances"]
        )
        ids, dists = res["ids"][0], res["distances"][0]
        return [VectorHit(chunk_id=i, similarity=1.0 - d) for i, d in zip(ids, dists)]

    def delete_document(self, document_id: str) -> None:
        self._collection.delete(where={"document_id": document_id})

    def count(self) -> int:
        return self._collection.count()

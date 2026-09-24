# System Architecture

## 1. High-level view

```mermaid
flowchart LR
    dev["Developers (~100)<br/>CLI / IDE / internal apps"] -->|HTTPS + X-API-Key / SSO| api

    subgraph platform["Knowledge Platform (FastAPI)"]
        api["API layer<br/>/v1/documents · /v1/query · /v1/llm/chat<br/>auth · validation · error envelope · request IDs"]
        search["Search service<br/>embed query → vector + BM25 → RRF → rerank"]
        gw["AI Gateway<br/>allow-list · rate limit · retries · fallbacks · usage log"]
        worker["Ingestion workers<br/>(DB-backed queue, leases, retries)"]
        api --> search
        api --> gw
        api -. "enqueue (row status=queued)" .-> worker
        search -. "generate_answer" .-> gw
    end

    subgraph models["Local ML models (CPU)"]
        emb["Embedder<br/>BAAI/bge-small-en-v1.5 (384-d)"]
        rr["Reranker<br/>cross-encoder"]
        ocr["OCR<br/>RapidOCR (PP-OCR, ONNX)"]
    end

    subgraph storage["Storage"]
        db[("Relational DB<br/>SQLite (dev) / Postgres (prod)<br/>documents · chunks · embeddings registry<br/>query_logs · llm_usage · FTS5 keyword index")]
        vdb[("Vector store<br/>Chroma HNSW (dev) / pgvector (prod)")]
        blob[("File storage<br/>local disk (dev) / S3 (prod)")]
    end

    worker --> ocr
    worker --> emb
    worker --> vdb
    worker --> db
    worker --> blob
    search --> emb
    search --> rr
    search --> vdb
    search --> db
    api --> db
    api --> blob
    gw --> llm["Anthropic Claude API<br/>(claude-opus-5)"]
    gw --> db
```

The relational DB is the **source of truth**. The vector store is a **derived index** that
can always be rebuilt from `chunks`. This one decision makes deletes, retries and
embedding-model migrations safe.

## 2. Ingestion flow (`POST /v1/documents`)

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant A as API
    participant S as File storage
    participant D as DB
    participant W as Worker
    participant V as Vector store

    C->>A: multipart upload (file, tags, metadata)
    A->>A: validate type / size / magic bytes, sha256
    A->>D: duplicate by sha256? → 200 + existing doc
    A->>S: put raw file
    A->>D: INSERT documents(status=queued)
    A-->>C: 202 Accepted + Location: /v1/documents/{id}
    A--)W: notify (wake-up signal)
    W->>D: claim job (compare-and-set UPDATE, lease=10 min)
    W->>S: read file
    W->>W: extract (text layer or OCR) → chunk (AST / heading / recursive)
    W->>W: embed chunks (batched)
    W->>V: upsert vectors (deterministic chunk IDs)
    W->>D: ONE txn: chunks + embeddings + FTS rows + status=ready
    C->>A: GET /v1/documents/{id}
    A-->>C: status=ready, chunk_count, page_count
```

Failure handling in the worker:

| Failure | Behaviour |
|---|---|
| Unsupported or empty or corrupt file, no extractable text | `failed` right away with a readable `error` (retrying can't help) |
| Transient error (embedding OOM, vector store down, DB locked) | `attempts += 1`, re-queued with exponential backoff; `failed` after `KP_MAX_INGEST_ATTEMPTS` |
| Worker crash mid-job | Lease (`locked_until`) expires and another worker re-claims the job. Nothing is lost on restart |
| Crash after vectors were written but before the SQL commit | Retry upserts the same deterministic IDs. Orphan vectors are never served, because query results are hydrated from SQL and filtered to `status=ready` |
| Document deleted while it is being processed | The final transaction re-checks status, discards the vectors it wrote, and leaves the purge to the delete path |

## 3. Query flow (`POST /v1/query`)

```mermaid
flowchart LR
    q[query + filters] --> f["resolve filters in SQL<br/>(tags, metadata, type, language, dates)<br/>→ allowed document IDs"]
    f --> e["embed query<br/>(bge query prefix, LRU cached)"]
    e --> v["vector top-N<br/>(HNSW cosine, doc-ID pre-filter)"]
    f --> k["BM25 top-N<br/>(FTS5, doc-ID pre-filter)"]
    v --> r["Reciprocal Rank Fusion"]
    k --> r
    r --> h["hydrate from SQL<br/>drop non-ready / deleted"]
    h --> x["cross-encoder rerank top-M"]
    x --> t["top-K results<br/>+ score breakdown"]
    t -. generate_answer .-> g["AI Gateway → Claude<br/>grounded answer with [n] citations"]
    t --> l[(query_logs)]
```

## 4. Delete flow (`DELETE /v1/documents/{id}`)

1. **Soft delete** (synchronous, one atomic UPDATE): `status=deleted`, `deleted_at=now`.
   From this moment, the document is invisible to every query and listing, because all
   read paths filter on `status`.
2. **Hard purge** (async, or inline with `?hard=true`): delete vectors → one SQL
   transaction deleting FTS rows, embeddings, chunks, tags and the document row → delete
   the raw file.
3. **Partial failure**: every purge step is idempotent. If the vector store is down, the
   document stays soft-deleted (still invisible), `purge_attempts` is incremented, and the
   worker retries with exponential backoff. Documents that exhaust their retries appear in
   `/ready` as `purges_needing_attention` for an operator.

**Why both?** A soft delete alone leaves data behind: storage grows, and "delete" requests
(e.g. accidentally uploaded secrets or PII) must mean deleted. A synchronous hard delete
alone spans two systems (DB + vector store) with no shared transaction, so a failure
halfway leaves a document that is half-searchable. Soft-then-purge gives an **immediate,
atomic user-visible effect** and **eventually-complete physical removal** with retries.

## 5. AI Gateway

All LLM traffic goes through `app/gateway.py`, both the RAG answer step and the
developer-facing `POST /v1/llm/chat`:

- **Credentials**: the provider key lives only on the server. Developers use platform keys.
- **Model allow-list** (`KP_LLM_ALLOWED_MODELS`) and a default model (`claude-opus-5`).
- **Per-user rate limiting** (token bucket; 429 with `retry_after_s`).
- **Resilience**: SDK retries with backoff on 408/409/429/5xx, request timeout, and
  server-side refusal fallbacks (`fallbacks="default"`).
- **Accounting**: every call is written to `llm_usage` (user, model, tokens, latency,
  status) for cost attribution and auditing.
- **Graceful degradation**: if the LLM is unavailable, `/v1/query` still returns the
  retrieval results, with `answer.error` set.

## 6. Code map

| Path | Responsibility |
|---|---|
| `app/main.py` | App factory, lifespan (model warm-up, worker start/stop), request-ID middleware |
| `app/api/` | HTTP routers: documents, query, llm gateway, health |
| `app/ingestion/extractors.py` | PDF text layer + OCR fallback, header/footer stripping, text decoding |
| `app/ingestion/chunkers.py` | Recursive prose splitter, markdown-by-heading, Python AST chunker, generic code |
| `app/ingestion/pipeline.py` | Ingest + purge pipelines (consistency rules live here) |
| `app/worker.py` | DB-backed job queue: claiming, leases, retries, backoff |
| `app/retrieval/` | Search service (hybrid + RRF + rerank), BM25 index, reranker, caches |
| `app/embeddings.py`, `app/vectorstore.py` | Swappable embedding and vector-store interfaces |
| `app/gateway.py` | AI Gateway |
| `app/db/` | SQLAlchemy models and engine setup |

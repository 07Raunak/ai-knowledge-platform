# API Specification

Base URL: `http://localhost:8000`. Interactive OpenAPI docs: **`/docs`** (Swagger UI) and
`/openapi.json`.

## Conventions

- **Versioning**: all business endpoints are under `/v1`.
- **Auth**: `X-API-Key: <key>` header. Keys map to user IDs via `KP_API_KEYS="key1:alice,key2:bob"`.
  When `KP_API_KEYS` is empty (local dev), auth is disabled and the caller is `anonymous`.
  In production this is replaced by SSO (OIDC JWT from the company IdP).
- **Request tracing**: send `X-Request-ID` or one is generated. It is echoed in the
  response header and in every log line.
- **Errors**: always the same envelope:

```json
{ "error": { "code": "unsupported_file_type", "message": "Unsupported file type for 'a.exe'",
             "details": { "supported_extensions": [".c", ".cpp", "..."] }, "request_id": "4f1c2a9b0d3e7a11" } }
```

| HTTP | `code` | When |
|---|---|---|
| 400 | `bad_request` | Semantically invalid request (e.g. reindex while processing, model not allowed) |
| 401 | `unauthorized` | Missing or invalid API key |
| 404 | `not_found` | Unknown or purged document |
| 413 | `payload_too_large` | File > `KP_MAX_UPLOAD_MB` (default 25 MB) |
| 415 | `unsupported_file_type` | Extension not supported, or `.pdf` without PDF magic bytes |
| 422 | `validation_error` | Schema validation failed (`details` lists fields) |
| 429 | `rate_limited` | Per-user LLM rate limit (`details.retry_after_s`) or upstream 429 |
| 502 | `upstream_error` | LLM provider error |
| 503 | `service_unavailable` | LLM gateway disabled or not configured |

---

## Documents

### `POST /v1/documents`: upload

`multipart/form-data`

| Field | Type | Required | Notes |
|---|---|---|---|
| `file` | file | yes | `.pdf .md .markdown .txt .rst` or code: `.py .js .jsx .ts .tsx .java .go .rb .rs .c .h .cpp .hpp .cs .php .kt .scala .swift .sql .sh .yaml .yml .toml .json` |
| `tags` | string | no | Comma-separated, lowercase `[a-z0-9_.:-]`, max 20 |
| `metadata` | string (JSON) | no | Flat object of scalars, e.g. `{"team":"platform","repo":"badger"}` |

**202 Accepted**: processing is asynchronous. The `Location` header points at the status URL.

```bash
curl -X POST localhost:8000/v1/documents \
  -F "file=@samples/Source_Code_Sample.py" \
  -F "tags=proxy,scraping" -F 'metadata={"team":"platform"}'
```
```json
{
  "document": {
    "id": "0b0c7f7e-3f7a-4c52-a2f5-7f1a1d8e8f10",
    "filename": "Source_Code_Sample.py",
    "file_type": "code",
    "language": "python",
    "size_bytes": 7366,
    "sha256": "…",
    "status": "queued",
    "error": null,
    "attempts": 0,
    "chunk_count": 0,
    "page_count": null,
    "embedding_model": null,
    "tags": ["proxy", "scraping"],
    "metadata": {"team": "platform"},
    "uploaded_by": "anonymous",
    "created_at": "2026-09-24T15:40:02.114Z",
    "updated_at": "2026-09-24T15:40:02.114Z",
    "processed_at": null,
    "deleted_at": null
  },
  "duplicate": false,
  "status_url": "/v1/documents/0b0c7f7e-3f7a-4c52-a2f5-7f1a1d8e8f10"
}
```

**200 OK** with `"duplicate": true` if identical bytes (same SHA-256) are already
ingested. Nothing is re-processed, so uploads are idempotent.

### `GET /v1/documents/{id}`: status and metadata

Returns the `document` object above. `status` moves through
`queued → processing → ready` or `failed` (with `error`), and `deleted` after a delete.

### `GET /v1/documents`: list

Query: `status`, `file_type`, `tag`, `limit` (1-100, default 20), `offset`.
Deleted documents are excluded unless `status=deleted`.

```json
{ "items": [ { "id": "…", "filename": "…", "status": "ready", "...": "..." } ], "total": 2, "limit": 20, "offset": 0 }
```

### `GET /v1/documents/{id}/chunks`: inspect chunks

Query: `limit` (≤200), `offset`. Returns `chunk_index`, `content`, `token_count`,
`chunk_type`, `section`, `page_start/page_end` or `start_line/end_line`.

### `DELETE /v1/documents/{id}`: delete

| Query param | Default | Meaning |
|---|---|---|
| `hard` | `false` | Also attempt the physical purge synchronously |

- **202**: soft-deleted (invisible to search immediately); purge scheduled or retrying.
- **200**: `hard=true` and the purge completed (`purged: true`).
- **404**: unknown document. Deleting an already soft-deleted document returns 202 again (idempotent).

```json
{ "id": "…", "status": "deleted", "purged": false,
  "message": "Document hidden from search immediately; physical purge scheduled" }
```

### `POST /v1/documents/{id}/reindex`

Re-runs extraction, chunking and embedding (e.g. after changing the chunker or the embedding
model). **202** with the document in `queued` state.

---

## Query

### `POST /v1/query`

```json
{
  "query": "What happens when a proxy request fails?",
  "top_k": 5,
  "filters": {
    "document_ids": null,
    "file_types": ["code"],
    "languages": ["python"],
    "tags": ["proxy"],
    "metadata": {"team": "platform"},
    "uploaded_by": null,
    "filename_contains": null,
    "created_after": null,
    "created_before": null
  },
  "rerank": null,
  "hybrid": null,
  "min_score": null,
  "generate_answer": false
}
```

| Field | Default | Notes |
|---|---|---|
| `query` | required | 1-2000 chars |
| `top_k` | 5 | 1-50 |
| `filters` | none | All conditions AND-ed. `tags` is any-of. `metadata` is exact match per key |
| `rerank` / `hybrid` | server default (on) | Per-request override for experiments |
| `min_score` | none | Drop results below this score |
| `generate_answer` | false | Also produce a grounded answer through the AI Gateway |

**200 OK**: results are ranked. `score` is the value they are ordered by: cross-encoder
relevance (0-1) when reranked, otherwise the fusion score. `scores` shows how each
retriever ranked the chunk.

```json
{
  "query_id": "7d8e…",
  "query": "What happens when a proxy request fails?",
  "results": [
    {
      "rank": 1,
      "score": 0.97,
      "chunk_id": "…",
      "document_id": "…",
      "filename": "Source_Code_Sample.py",
      "file_type": "code",
      "language": "python",
      "chunk_index": 5,
      "section": "DecayProxyRotator.report_failure",
      "page_start": null, "page_end": null,
      "start_line": 57, "end_line": 67,
      "content": "    def report_failure(self, proxy):\n        \"\"\"Increase penalty multiplier and reset score to zero.\"\"\" …",
      "scores": { "vector_similarity": 0.71, "vector_rank": 1, "keyword_rank": 2, "rrf": 0.0325, "rerank": 0.97 }
    }
  ],
  "answer": {
    "text": "When a proxy fails, `report_failure` resets its score to 0, increments failure_count and raises its penalty_factor by 2.0, so it recovers more slowly [1].",
    "model": "claude-opus-5",
    "citations": [1],
    "error": null
  },
  "reranked": true,
  "hybrid": true,
  "cache_hit": false,
  "timings_ms": { "embed": 32.0, "vector_search": 3.6, "keyword_search": 1.2, "rerank": 1850.0, "total": 1887.0, "request_total": 1887.4 }
}
```

If the LLM is unavailable, `answer.text` is `null` and `answer.error` explains why. The
retrieval results are still returned.

---

## AI Gateway

### `POST /v1/llm/chat`

```json
{
  "messages": [{ "role": "user", "content": "Summarize what RRF does in two sentences." }],
  "system": "You are a concise senior engineer.",
  "model": "claude-opus-5",
  "max_tokens": 1024
}
```
```json
{ "model": "claude-opus-5", "content": "…", "stop_reason": "end_turn",
  "input_tokens": 31, "output_tokens": 58, "latency_ms": 2140.3, "fallback_used": false }
```

`model` must be in `KP_LLM_ALLOWED_MODELS`. Calls are rate-limited per user
(`KP_LLM_RATE_LIMIT_PER_MINUTE`) and recorded in `llm_usage`.

---

## Operations

| Endpoint | Purpose |
|---|---|
| `GET /health` | Liveness: process is up |
| `GET /ready` | Readiness: DB and vector store reachable, documents by status (queue depth), purges needing attention, active models. 503 if a dependency is down |

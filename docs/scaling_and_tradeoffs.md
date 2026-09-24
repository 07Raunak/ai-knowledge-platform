# Scaling Strategy, Assumptions and Trade-offs

## Assumptions

- **Users**: ~100 internal developers. Peak maybe 5-10 queries/s. Uploads are bursty but
  low volume (tens to hundreds per day).
- **Corpus**: 10³-10⁵ documents → roughly 10⁴-10⁶ chunks. Most files are under 25 MB.
- **Latency targets**: retrieval p95 < 500 ms without answer generation (measured locally:
  ~40-100 ms for embed + vector + BM25; the CPU cross-encoder adds 1-2 s, see below). Ingestion is
  asynchronous, so seconds to minutes are acceptable. OCR'd PDFs take ~9 s/page on CPU.
- **Trust boundary**: internal network, SSO-authenticated users. All developers can read
  all documents (no per-document ACLs in this version; see "Next steps").
- **Data sensitivity**: embeddings and text are computed locally (no document content
  leaves the network) *except* when a user explicitly asks for `generate_answer` or
  calls `/v1/llm/chat`. Only then are the retrieved chunks sent to the LLM provider.

## What runs where today (single node)

| Concern | Choice | Why it's enough at this scale |
|---|---|---|
| API | FastAPI + uvicorn | Sync handlers run in a thread pool, so model inference never blocks the event loop |
| Metadata | SQLite (WAL) | One writer plus concurrent readers, zero ops. Same SQLAlchemy models run on Postgres by changing `KP_DATABASE_URL` |
| Vectors | Chroma (embedded, HNSW) | Millions of 384-d vectors fit in RAM, and it supports metadata filters |
| Queue | The `documents` table (leases + compare-and-set) | Durable, survives restarts, and works across processes without a broker |
| Models | bge-small (33 M params), MiniLM cross-encoder, RapidOCR | All CPU-friendly. No GPU or external API is needed for retrieval |

## Scaling path

**Stage 1 (now, ~100 users).** One API process with 2 worker threads. Handles the
stated load comfortably. Measured locally: code-file ingestion ~6 s, 22-page OCR PDF
~3 min. Search: ~40-100 ms for embed + hybrid retrieval, **plus 1-2 s for cross-encoder
reranking on this laptop CPU** (~80 ms per 512-token pair). `"rerank": false` skips it per
request. In production the reranker runs on a GPU or as an ONNX/int8 export (typically
<100 ms for 12 pairs), or behind its own autoscaled inference service.

**Stage 2 (~1k users / 10⁶ chunks). Split the roles, share the state.**

```
          ┌──────────── load balancer ────────────┐
          ▼                    ▼                   ▼
     API replica 1        API replica 2       API replica N      (stateless, autoscaled on CPU/p95)
          │                    │                   │
          └──────┬─────────────┴──────────┬────────┘
                 ▼                        ▼
   PostgreSQL + pgvector (primary      Redis (result/embedding cache,
   + read replica for search)          rate-limit buckets, corpus_version)
                 ▲
   Ingestion workers (separate deployment, autoscaled on queue depth;
   OCR/embedding are the CPU hogs, so they scale independently of the API)
                 │
   S3 / blob storage for raw files
```

- **Postgres + pgvector** replaces SQLite + Chroma. Vectors, chunks and documents then
  share one ACID store, so the ingest commit and the purge each become one transaction,
  and the "vectors first, then SQL" ordering is no longer needed.
- **Workers** become their own deployment. The claim query is already safe for many
  concurrent workers (`UPDATE … WHERE status='queued'` compare-and-set, or
  `SELECT … FOR UPDATE SKIP LOCKED` on Postgres). Optionally add SQS/Redis Streams for
  push-based wake-ups; the state machine stays the same.
- **Caches and rate limits** move to Redis (same keys), so all replicas share them.
- **Embeddings on GPU** or via a dedicated embedding service (TEI / Triton) with dynamic
  batching once ingestion volume grows.

**Stage 3 (10⁷+ chunks, many teams).**

- Dedicated vector DB (Qdrant / Milvus / OpenSearch k-NN) sharded by `document_id` or by
  team, with payload indexes for filters. Or partition pgvector tables by hash of
  `document_id`.
- Quantization (int8 / binary + rescoring) to cut vector RAM 4-32×.
- Query logs streamed to the warehouse. Monthly partitions dropped from Postgres.
- Per-team collections when isolation or ACLs demand it.

## Reliability

| Risk | Mitigation |
|---|---|
| Worker crash mid-ingestion | Lease expiry + re-claim, deterministic chunk IDs (idempotent upsert) |
| Transient dependency failure | Bounded retries with exponential backoff, then `failed` with a readable error |
| Bad or poison files | Validated at upload (type, size, magic bytes). Permanent extraction errors are not retried |
| Partial delete | Soft delete first (atomic), idempotent purge retried. Reads always filter on `status`, so partial states are never visible |
| Vector/SQL divergence | SQL is the source of truth. Results are hydrated from SQL, and the vector index is rebuildable (`reindex`) |
| LLM outage / refusal | Retrieval still returns. SDK retries with backoff, server-side refusal fallbacks, per-user rate limits |
| Observability | Request IDs, structured logs (`KP_LOG_JSON=true`), `/health` + `/ready` (queue depth, stuck purges), `query_logs` + `llm_usage` tables for analytics and cost. Next step: Prometheus metrics and OpenTelemetry traces |

## Trade-offs

| Decision | Chosen | Alternative | Why |
|---|---|---|---|
| Embedding model | `bge-small-en-v1.5`, local | OpenAI / Voyage embedding API | Free, private (content never leaves the network), fast on CPU. Larger or hosted models retrieve better; swapping is a config change + `reindex`, and the new model's vectors go into a separate collection |
| Retrieval | Hybrid (vector + BM25) + RRF | Pure vector | Developers search for identifiers (`report_failure`, error codes) that dense models blur. RRF needs no score calibration |
| Reranking | Cross-encoder on the fused top-12 | None / LLM reranker | Big precision gain (it fixed the code-query ordering in the demo). On a laptop CPU it costs 1-2 s per query, which is the main latency cost today; GPU/ONNX brings it to <100 ms. LLM reranking is better still but costs seconds and money per query |
| Chunking | Structure-aware (AST for Python, headings for markdown, pages for PDF) | Fixed-size windows | Chunks align with meaning (one method = one chunk) and can be cited by line or page. Costs a chunker per format; unknown code languages fall back to line windows |
| Chunk size | ~350 tokens, 50 overlap | Larger chunks | Stays under bge's 512-token limit with headroom. Small enough for precise hits, big enough to hold a full paragraph or method |
| OCR | RapidOCR at 150 DPI + 300 DPI rescue of low-confidence lines | Tesseract / cloud OCR / always 300 DPI | pip-installable (no system deps), good accuracy. The rescue pass fixes garbled lines for a fraction of full high-DPI cost |
| Queue | DB table as queue | Celery + Redis / SQS | One less moving part, transactional with document state. Swap in a broker when throughput demands it |
| Delete | Soft delete → async purge | Synchronous hard delete | Atomic user-visible effect across two stores, with retries on partial failure. Cost: data physically lingers until the purge succeeds (seconds normally) |
| Dedup | Content SHA-256 | None | Idempotent uploads and no duplicate results. A changed file is a new version (new document) |
| Consistency | SQL source of truth, vector index derived | Dual-write treated as equal | Clear recovery story: any divergence is healed by re-deriving from SQL |
| Caching | In-process LRU + corpus version | Redis from day one | Zero ops at one node. Invalidation is correct by construction. Moves to Redis at Stage 2 |

## Known limitations / next steps

1. **Access control**: add document-level ACLs (owner / team / public) and apply them as
   a mandatory filter in `resolve_document_filter`.
2. **Document versioning**: keep `(source_uri, version)` so re-uploading a changed file
   supersedes the old version instead of creating a sibling.
3. **Code-aware reranking**: the default MS-MARCO cross-encoder is trained on web
   passages. A small benchmark on the sample code file (rank of the correct method for 4
   queries) gave: MiniLM `[4,3,1,1]` → `[4,1,1,1]` after prefixing chunks with
   `file | symbol` (now the default), and `BAAI/bge-reranker-base` + header `[1,1,1,1]`
   at ~6× the CPU cost. That model is a one-line switch (`KP_RERANKER_MODEL`) when a GPU
   is available. A code-tuned embedding model (e.g. `jina-embeddings-v2-base-code`) would
   also help.
4. **Evaluation**: a labelled query set (from `query_logs` + feedback) with recall@k / MRR
   gates in CI before changing models or chunkers.
5. **Streaming answers** (SSE) for `generate_answer` to cut time-to-first-token.
6. **OCR throughput**: parallelize pages across a process pool, or use a GPU OCR service
   for large scanned archives.

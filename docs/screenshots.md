# Proof of Execution: Screenshots

Taken from a live run against the local server (`uvicorn app.main:app`), using the
Swagger UI at `http://localhost:8000/docs`, starting from an empty system.
Video of the same run: [demo_video.mp4](https://drive.google.com/file/d/1mw5StwI4WUdNazgXEfe4CCCfvPfif6PT/view?usp=sharing).

## 0. Empty system

`GET /ready`: database and vector store are up, no documents yet.

![ready empty](screenshots/02_ready_empty.png)

## 1. Upload the two task files (`POST /v1/documents`)

`Source_Code_Sample.py` is accepted with **202**, `status: queued`, with tags and metadata stored:

![upload code](screenshots/03_upload_code_202.png)

`Knowledge_Base_Sample.pdf` is accepted with **202**:

![upload pdf](screenshots/04_upload_pdf_202.png)

## 2. Asynchronous processing (`GET /v1/documents/{id}`)

The code file is `ready`: **15 chunks**, embedded with `BAAI/bge-small-en-v1.5`:

![code ready](screenshots/05_code_ready.png)

The PDF is `ready`: **22 pages OCR'd** (the PDF has no text layer) into **16 chunks**:

![pdf ready](screenshots/08_pdf_ready.png)

## 3. Chunking (`GET /v1/documents/{id}/chunks`)

The code is chunked per function or method, with line numbers:

![code chunks](screenshots/07_code_chunks_methods.png)

The PDF text is recovered by OCR, and each chunk keeps its page range:

![pdf chunks](screenshots/09_pdf_chunks.png)

## 4. Semantic search (`POST /v1/query`)

| # | Question | Rank-1 result |
|---|---|---|
| 1 | What is AI orchestration? | PDF p.3-4, the definition |
| 2 | What productivity lift did the McKinsey report find? | PDF p.11-12, "25% productivity lift" |
| 3 | What is MCP and why does it matter? | PDF p.14-16, the MCP section |
| 4 | What happens when a proxy request fails? | `DecayProxyRotator.report_failure` L57-67 |
| 5 | How are UAs rotated / blocked by a CAPTCHA? | `UAFreshnessRotator.report_block` L127-136 |
| 6 | Filtered: `languages=[python], tags=[proxy]` | `DecayProxyRotator.report_failure` (code only) |

![q1](screenshots/10_query_1.png)
![q2](screenshots/11_query_2.png)
![q3](screenshots/12_query_3.png)
![q4](screenshots/13_query_4.png)
![q5](screenshots/14_query_5.png)
![q6](screenshots/15_query_6.png)

## 5. Delete (`DELETE /v1/documents/{id}`)

A throwaway runbook is found by search:

![before](screenshots/16_delete_before.png)

It is deleted. The soft delete takes effect immediately and the physical purge runs in the background:

![delete](screenshots/17_delete_202.png)

The same search no longer returns it:

![after](screenshots/18_delete_after_search.png)

After the purge, the document is gone (**404**):

![404](screenshots/19_delete_404.png)

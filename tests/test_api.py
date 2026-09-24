import json

from app.db.models import Document
from tests.conftest import SAMPLES, wait_for_status


def upload(client, name, data, **form):
    return client.post("/v1/documents", files={"file": (name, data)}, data=form)


def test_upload_process_query_delete_lifecycle(client):
    code = (SAMPLES / "Source_Code_Sample.py").read_bytes()
    r = upload(client, "Source_Code_Sample.py", code, tags="proxy,scraping", metadata=json.dumps({"team": "platform"}))
    assert r.status_code == 202, r.text
    body = r.json()
    assert body["duplicate"] is False and r.headers["Location"] == body["status_url"]
    doc_id = body["document"]["id"]

    doc = wait_for_status(client, doc_id)
    assert doc["status"] == "ready", doc
    assert doc["chunk_count"] > 5 and doc["language"] == "python"
    assert doc["tags"] == ["proxy", "scraping"]

    res = client.post("/v1/query", json={"query": "report_failure penalty factor", "top_k": 3}).json()
    assert res["results"][0]["filename"] == "Source_Code_Sample.py"
    assert res["results"][0]["section"] == "DecayProxyRotator.report_failure"
    assert [r["rank"] for r in res["results"]] == [1, 2, 3]

    # metadata filters: matching and non-matching
    ok = client.post("/v1/query", json={"query": "proxy", "filters": {"tags": ["proxy"], "metadata": {"team": "platform"}}})
    assert ok.json()["results"]
    none = client.post("/v1/query", json={"query": "proxy", "filters": {"file_types": ["pdf"]}})
    assert none.json()["results"] == []

    # identical re-upload is de-duplicated
    dup = upload(client, "copy.py", code)
    assert dup.status_code == 200 and dup.json()["duplicate"] is True
    assert dup.json()["document"]["id"] == doc_id

    # delete: invisible immediately, physically purged afterwards
    d = client.delete(f"/v1/documents/{doc_id}")
    assert d.status_code == 202 and d.json()["status"] == "deleted"
    after = client.post("/v1/query", json={"query": "report_failure penalty factor"}).json()
    assert after["results"] == []
    assert wait_for_status(client, doc_id, statuses=("gone",))["status"] == "gone"
    c = client.app.state.container
    assert c.vectors.count() == 0


def test_hard_delete_is_synchronous(client):
    r = upload(client, "notes.md", b"# Notes\n\nThe cache TTL is five minutes.")
    doc_id = r.json()["document"]["id"]
    wait_for_status(client, doc_id)
    d = client.delete(f"/v1/documents/{doc_id}", params={"hard": "true"})
    assert d.status_code == 200 and d.json()["purged"] is True
    assert client.get(f"/v1/documents/{doc_id}").status_code == 404


def test_purge_retries_after_vector_store_failure(client, monkeypatch):
    r = upload(client, "a.txt", b"vector store failure drill content")
    doc_id = r.json()["document"]["id"]
    wait_for_status(client, doc_id)
    c = client.app.state.container

    calls = {"n": 0}
    real_delete = c.vectors.delete_document

    def flaky_delete(document_id):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectionError("vector store unavailable")
        return real_delete(document_id)

    monkeypatch.setattr(c.vectors, "delete_document", flaky_delete)
    d = client.delete(f"/v1/documents/{doc_id}", params={"hard": "true"})
    assert d.status_code == 202 and d.json()["purged"] is False  # partial failure reported
    # still hidden from search while the purge is pending
    assert client.post("/v1/query", json={"query": "vector store failure drill"}).json()["results"] == []
    # the worker retries with backoff and completes the purge
    assert wait_for_status(client, doc_id, statuses=("gone",))["status"] == "gone"
    assert calls["n"] >= 2


def test_failed_extraction_marks_document_failed(client):
    r = upload(client, "empty.txt", b"   \n  ")
    doc = wait_for_status(client, r.json()["document"]["id"])
    assert doc["status"] == "failed" and "empty" in doc["error"].lower()


def test_transient_ingest_error_is_retried(client, monkeypatch):
    c = client.app.state.container
    real = c.embedder.embed_documents
    calls = {"n": 0}

    def flaky(texts):
        calls["n"] += 1
        if calls["n"] == 1:
            raise TimeoutError("embedding service timeout")
        return real(texts)

    monkeypatch.setattr(c.embedder, "embed_documents", flaky)
    r = upload(client, "retry.txt", b"retry me please")
    doc = wait_for_status(client, r.json()["document"]["id"])
    assert doc["status"] == "ready" and doc["attempts"] == 1


def test_validation_errors_use_error_envelope(client):
    r = upload(client, "virus.exe", b"MZ")
    assert r.status_code == 415 and r.json()["error"]["code"] == "unsupported_file_type"
    r = upload(client, "fake.pdf", b"not a pdf")
    assert r.status_code == 415
    r = client.post("/v1/query", json={"query": "   "})
    assert r.status_code == 422 and r.json()["error"]["code"] == "validation_error"
    r = client.post("/v1/query", json={"query": "x", "filters": {"unknown": 1}})
    assert r.status_code == 422
    assert client.get("/v1/documents/does-not-exist").status_code == 404


def test_query_logs_are_recorded(client):
    client.post("/v1/query", json={"query": "anything at all"})
    from app.db.models import QueryLog

    with client.app.state.container.sessions() as s:
        logs = s.query(QueryLog).all()
    assert len(logs) == 1 and logs[0].query_text == "anything at all"


def test_generate_answer_degrades_gracefully_when_llm_disabled(client):
    r = upload(client, "kb.txt", b"The on-call rotation changes every Monday at 10am.")
    wait_for_status(client, r.json()["document"]["id"])
    res = client.post("/v1/query", json={"query": "when does on-call rotate", "generate_answer": True}).json()
    assert res["results"]
    assert res["answer"]["text"] is None and "disabled" in res["answer"]["error"]


def test_api_key_auth(tmp_path, settings):
    from fastapi.testclient import TestClient

    from app.main import create_app

    secured = settings.model_copy(update={"api_keys": "secret-1:alice"})
    with TestClient(create_app(secured)) as c:
        assert c.post("/v1/query", json={"query": "x"}).status_code == 401
        assert c.post("/v1/query", json={"query": "x"}, headers={"X-API-Key": "wrong"}).status_code == 401
        r = upload(c, "a.txt", b"hello", )
        assert r.status_code == 401
        r = c.post("/v1/documents", files={"file": ("a.txt", b"hello")}, headers={"X-API-Key": "secret-1"})
        assert r.status_code == 202 and r.json()["document"]["uploaded_by"] == "alice"

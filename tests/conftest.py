import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app

SAMPLES = Path(__file__).resolve().parent.parent / "samples"


@pytest.fixture()
def settings(tmp_path) -> Settings:
    # Offline + fast: hashing embedder, no reranker model, no OCR, LLM disabled.
    return Settings(
        environment="test",
        data_dir=tmp_path,
        embedding_backend="hash",
        reranker_enabled=False,
        ocr_enabled=False,
        llm_provider="disabled",
        worker_poll_interval_s=0.05,
        retry_base_delay_s=0.01,
        api_keys="",
    )


@pytest.fixture()
def client(settings):
    with TestClient(create_app(settings)) as c:
        yield c


def wait_for_status(client, doc_id: str, statuses=("ready", "failed"), timeout: float = 30.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = client.get(f"/v1/documents/{doc_id}")
        if r.status_code == 404 and "gone" in statuses:
            return {"status": "gone"}
        body = r.json()
        if body.get("status") in statuses:
            return body
        time.sleep(0.05)
    raise AssertionError(f"document {doc_id} did not reach {statuses}")

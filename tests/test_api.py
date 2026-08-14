import io

from fastapi.testclient import TestClient

from app.main import app


def test_health():
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_ingest_and_query():
    client = TestClient(app)
    content = (
        b"The refund window is 14 days from purchase. "
        b"After that charges are non-refundable."
    )
    files = {"file": ("faq.txt", io.BytesIO(content), "text/plain")}
    r = client.post("/ingest", files=files)
    assert r.status_code == 200, r.text
    assert r.json()["num_chunks"] >= 1

    r = client.post("/query", json={"question": "How long is the refund window?"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["grounded"] is True
    assert body["citations"]


def test_ingest_unsupported_format():
    client = TestClient(app)
    files = {"file": ("bad.xyz", io.BytesIO(b"data"), "application/octet-stream")}
    r = client.post("/ingest", files=files)
    assert r.status_code == 415

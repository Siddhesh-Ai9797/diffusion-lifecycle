import pytest
from fastapi.testclient import TestClient

serve = pytest.importorskip("src.serve")
client = TestClient(serve.app)


def test_health():
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["device"] in {"cpu", "cuda"}


def test_metrics_empty_before_requests():
    serve.request_log.clear()
    assert client.get("/metrics").json() == {"message": "No requests yet"}


def test_percentiles():
    values = list(range(1, 101))
    assert serve.compute_percentile(values, 50) in (50, 51)
    assert serve.compute_percentile(values, 99) >= 99

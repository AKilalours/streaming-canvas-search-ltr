from fastapi.testclient import TestClient

from app.main import app


def test_health():
    with TestClient(app) as c:
        r = c.get("/health")
        assert r.status_code == 200
        assert "ok" in r.json()


def test_search_returns_shape_even_without_artifacts():
    with TestClient(app) as c:
        r = c.post("/search", json={"query": "test query", "k": 5, "debug": True})
        assert r.status_code == 200
        data = r.json()
        assert data["query"] == "test query"
        assert data["k"] == 5
        assert "hits" in data

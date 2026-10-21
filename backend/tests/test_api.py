"""
Integration and API endpoint tests using FastAPI TestClient.
"""

from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "Book Search" in data["title"]


def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "readiness_details" in data


def test_dataset_stats_endpoint():
    response = client.get("/api/dataset/stats")
    assert response.status_code == 200
    data = response.json()
    assert "books_loaded" in data
    assert data["books_loaded"] > 200000


def test_indexing_stats_endpoint():
    response = client.get("/api/indexing/stats")
    assert response.status_code == 200
    data = response.json()
    assert "vocabulary_size" in data
    assert data["vocabulary_size"] > 0


def test_indexing_inspect_endpoint():
    response = client.get("/api/indexing/inspect?query=harry+potter")
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "harry potter"
    assert "potter" in data["processed_tokens"]
    assert data["total_candidate_documents"] > 0


def test_search_endpoint_valid():
    payload = {
        "query": "The Da Vinci Code",
        "model": "hybrid",
        "limit": 5
    }
    response = client.post("/api/search", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "The Da Vinci Code"
    assert len(data["results"]) > 0
    assert data["results"][0]["rank"] == 1


def test_search_endpoint_all_models():
    for model_name in ["tfidf", "bm25", "semantic", "hybrid"]:
        payload = {
            "query": "Harry Potter",
            "model": model_name,
            "limit": 3
        }
        response = client.post("/api/search", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "results" in data


def test_search_endpoint_invalid_model():
    payload = {
        "query": "Harry Potter",
        "model": "invalid_super_model",
        "limit": 5
    }
    response = client.post("/api/search", json=payload)
    assert response.status_code == 400


def test_evaluation_summary_endpoint():
    response = client.get("/api/evaluation/summary")
    assert response.status_code == 200
    data = response.json()
    assert "metrics_comparison" in data
    assert len(data["metrics_comparison"]) == 4


def test_evaluation_comparison_endpoint():
    response = client.get("/api/evaluation/comparison")
    assert response.status_code == 200
    data = response.json()
    assert "comparison" in data
    assert len(data["comparison"]) == 4


def test_evaluation_precision_recall_endpoint():
    response = client.get("/api/evaluation/precision-recall")
    assert response.status_code == 200
    data = response.json()
    assert "precision_recall_curves" in data
    assert "Hybrid RRF" in data["precision_recall_curves"]

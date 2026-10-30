"""Unit Tests for Developer REST API Endpoints."""

import json
from app.models import db, User


def test_api_predict_missing_token(client):
    """Ensure unauthenticated API calls return 401 Unauthorized."""
    res = client.post("/api/predict", json={"text": "Hello world"})
    assert res.status_code == 401
    data = res.get_json()
    assert data["code"] == "MISSING_API_KEY"


def test_api_predict_invalid_token(client):
    """Ensure invalid tokens return 403 Forbidden."""
    res = client.post(
        "/api/predict",
        headers={"X-API-Key": "totally-bogus-token-12345"},
        json={"text": "Hello world"},
    )
    assert res.status_code == 403
    data = res.get_json()
    assert data["code"] == "INVALID_API_KEY"


def test_api_predict_empty_text(client, test_user, app):
    """Ensure empty text payload is rejected with 422 Unprocessable Entity."""
    with app.app_context():
        user = db.session.get(User, test_user.id)
        api_key = user.api_key

    res = client.post(
        "/api/predict",
        headers={"X-API-Key": api_key},
        json={"text": "   "},
    )
    assert res.status_code == 422
    data = res.get_json()
    assert data["code"] == "EMPTY_TEXT"


def test_api_predict_success(client, test_user, app):
    """Ensure valid API call returns complete prediction structure."""
    with app.app_context():
        user = db.session.get(User, test_user.id)
        api_key = user.api_key

    payload = {
        "title": "Clinical Trial Results",
        "text": "A multi-center Phase III clinical trial confirmed that a novel vaccine achieved 85 percent efficacy against malaria in pediatric patients.",
        "domain": "nejm.org",
    }
    res = client.post(
        "/api/predict",
        headers={"X-API-Key": api_key},
        json=payload,
    )
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "success"
    assert "verdict" in data["data"]
    assert "confidence" in data["data"]
    assert "credibility_score" in data["data"]
    assert "rule_flags" in data["data"]
    assert "top_features" in data["data"]
    assert "disclaimer" in data["data"]


def test_api_history_success(client, test_user, app):
    """Ensure authenticated API caller can fetch paginated history."""
    with app.app_context():
        user = db.session.get(User, test_user.id)
        api_key = user.api_key

    res = client.get(
        "/api/history?page=1&per_page=5",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "success"
    assert "pagination" in data
    assert "records" in data
    assert isinstance(data["records"], list)
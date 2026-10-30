"""TruthLens Developer REST API Routes.

Endpoints:
- POST /api/predict: JSON classification & explainability (API token authenticated)
- GET /api/history: Retrieve past verification telemetry
- GET /api/docs: Interactive API documentation and specification
"""

from functools import wraps
from flask import Blueprint, request, jsonify, render_template
from app.models import db, User, CheckRecord
from ml.predict import predict

api_bp = Blueprint("api", __name__)


def require_api_key(f):
    """Validate per-user API key via X-API-Key header or Bearer token."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get("X-API-Key")
        if not api_key:
            auth_header = request.headers.get("Authorization")
            if auth_header and auth_header.startswith("Bearer "):
                api_key = auth_header.split(" ")[1]

        if not api_key:
            return jsonify({
                "status": "error",
                "code": "MISSING_API_KEY",
                "message": "Missing API token. Supply 'X-API-Key' header or 'Authorization: Bearer <token>'."
            }), 401

        user = User.query.filter_by(api_key=api_key).first()
        if not user or not user.is_active_account:
            return jsonify({
                "status": "error",
                "code": "INVALID_API_KEY",
                "message": "Invalid or deactivated API token."
            }), 403

        # Attach authenticated user to request context
        request.api_user = user
        return f(*args, **kwargs)

    return decorated_function


@api_bp.route("/predict", methods=["POST"])
@require_api_key
def api_predict():
    """Predict news credibility via JSON POST payload.

    Request Body:
        {
            "text": "Statement or article excerpt...",
            "domain": "optional-domain.com",
            "title": "Optional headline"
        }
    """
    data = request.get_json(silent=True)
    if not data or not isinstance(data, dict):
        return jsonify({
            "status": "error",
            "code": "BAD_REQUEST",
            "message": "Invalid JSON payload. Expected a JSON object with a 'text' property."
        }), 400

    text = data.get("text", "").strip()
    if not text:
        return jsonify({
            "status": "error",
            "code": "EMPTY_TEXT",
            "message": "The 'text' field cannot be empty."
        }), 422

    domain = data.get("domain", "").strip()
    title = data.get("title", "").strip()

    # Run inference
    result = predict(text, domain=domain if domain else None)

    # Persist record in user's history
    record = CheckRecord(
        user_id=request.api_user.id,
        title=title or text[:60] + "...",
        article_text=text,
        input_mode="api",
        domain=domain,
        verdict=result["verdict"],
        confidence=result["confidence"],
        credibility_score=result["credibility_score"],
        badge_color=result["badge_color"],
    )
    record.rule_flags = result.get("rule_flags", [])
    record.top_features = result.get("top_features", [])
    db.session.add(record)
    db.session.commit()

    return jsonify({
        "status": "success",
        "data": {
            "check_id": record.id,
            "verdict": result["verdict"],
            "confidence": result["confidence"],
            "credibility_score": result["credibility_score"],
            "probabilities": result["probabilities"],
            "rule_flags": result["rule_flags"],
            "top_features": result["top_features"],
            "disclaimer": result["disclaimer"],
        }
    }), 200


@api_bp.route("/history", methods=["GET"])
@require_api_key
def api_history():
    """Retrieve paginated audit history for the authenticated API key."""
    page = request.args.get("page", 1, type=int)
    per_page = min(50, request.args.get("per_page", 10, type=int))

    pagination = CheckRecord.query.filter_by(
        user_id=request.api_user.id
    ).order_by(CheckRecord.created_at.desc()).paginate(page=page, per_page=per_page, error_out=False)

    return jsonify({
        "status": "success",
        "pagination": {
            "page": pagination.page,
            "per_page": pagination.per_page,
            "total_items": pagination.total,
            "total_pages": pagination.pages,
        },
        "records": [r.to_dict() for r in pagination.items],
    }), 200


@api_bp.route("/docs")
def api_docs():
    """Interactive Swagger-style API documentation view."""
    return render_template("main/api_docs.html")


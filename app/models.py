"""TruthLens Database Models Module.

Defines:
- User (Roles, password hashing, API tokens, reset tokens)
- CheckRecord (History of news audits, predictions, scores, flags, XAI outputs)
- Feedback (Human-in-the-loop labels for admin review and model retraining)
- ActivityLog (Audit trail for security and admin monitoring)
"""

import datetime
import json
import secrets
from typing import Optional, List, Dict, Any
from flask_login import UserMixin
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash

db = SQLAlchemy()


class User(UserMixin, db.Model):
    """User account model with RBAC, secure hashing, and API authentication."""

    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    full_name = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(20), nullable=False, default="user")  # "user" or "admin"
    is_active_account = db.Column(db.Boolean, default=True, nullable=False)
    api_key = db.Column(db.String(64), unique=True, nullable=False, index=True)
    reset_token = db.Column(db.String(100), nullable=True)
    reset_token_expiry = db.Column(db.DateTime, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow, nullable=False)

    # Relationships
    checks = db.relationship("CheckRecord", backref="user", cascade="all, delete-orphan", lazy="dynamic", foreign_keys="CheckRecord.user_id")
    feedbacks = db.relationship("Feedback", backref="user", cascade="all, delete-orphan", lazy="dynamic", foreign_keys="Feedback.user_id")
    activities = db.relationship("ActivityLog", backref="user", cascade="all, delete-orphan", lazy="dynamic", foreign_keys="ActivityLog.user_id")

    def __init__(self, full_name: str, email: str, role: str = "user"):
        self.full_name = full_name.strip()
        self.email = email.strip().lower()
        self.role = role
        self.api_key = secrets.token_hex(24)
        self.created_at = datetime.datetime.utcnow()

    @property
    def is_active(self) -> bool:
        """Override Flask-Login property to honor account deactivation."""
        return self.is_active_account

    @property
    def is_admin(self) -> bool:
        """Check if user has administrative privileges."""
        return self.role == "admin"

    def set_password(self, password: str):
        """Hash and persist user password."""
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        """Verify candidate plaintext password against salt-hashed hash."""
        return check_password_hash(self.password_hash, password)

    def generate_new_api_key(self) -> str:
        """Regenerate user API token."""
        self.api_key = secrets.token_hex(24)
        return self.api_key

    def create_reset_token(self, expires_in_minutes: int = 30) -> str:
        """Generate a cryptographically secure token for password recovery."""
        token = secrets.token_urlsafe(32)
        self.reset_token = token
        self.reset_token_expiry = datetime.datetime.utcnow() + datetime.timedelta(minutes=expires_in_minutes)
        return token

    def verify_reset_token(self, token: str) -> bool:
        """Check token validity and expiration window."""
        if not self.reset_token or not self.reset_token_expiry:
            return False
        if self.reset_token != token:
            return False
        if datetime.datetime.utcnow() > self.reset_token_expiry:
            return False
        return True

    def clear_reset_token(self):
        """Clear token upon successful password update."""
        self.reset_token = None
        self.reset_token_expiry = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize user instance for API responses."""
        return {
            "id": self.id,
            "full_name": self.full_name,
            "email": self.email,
            "role": self.role,
            "is_active": self.is_active_account,
            "created_at": self.created_at.isoformat(),
        }


class CheckRecord(db.Model):
    """Stores full telemetry and explainability outputs for an article verification check."""

    __tablename__ = "check_records"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    title = db.Column(db.String(255), nullable=True)
    article_text = db.Column(db.Text, nullable=False)
    input_mode = db.Column(db.String(20), nullable=False, default="text")  # "text", "url", "file"
    url_source = db.Column(db.String(500), nullable=True)
    domain = db.Column(db.String(150), nullable=True)
    verdict = db.Column(db.String(20), nullable=False)  # "REAL", "FAKE", "UNCERTAIN"
    confidence = db.Column(db.Float, nullable=False)
    credibility_score = db.Column(db.Integer, nullable=False)  # 0 to 100
    badge_color = db.Column(db.String(20), nullable=False, default="secondary")
    rule_flags_json = db.Column(db.Text, nullable=True)
    top_features_json = db.Column(db.Text, nullable=True)
    highlighted_html = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow, nullable=False, index=True)

    # Relationship to user feedback
    feedback = db.relationship("Feedback", backref="check_record", uselist=False, cascade="all, delete-orphan")

    @property
    def rule_flags(self) -> List[Dict[str, Any]]:
        """Deserialize rule flags JSON."""
        return json.loads(self.rule_flags_json) if self.rule_flags_json else []

    @rule_flags.setter
    def rule_flags(self, val: List[Dict[str, Any]]):
        self.rule_flags_json = json.dumps(val)

    @property
    def top_features(self) -> List[Dict[str, Any]]:
        """Deserialize top features JSON."""
        return json.loads(self.top_features_json) if self.top_features_json else []

    @top_features.setter
    def top_features(self, val: List[Dict[str, Any]]):
        self.top_features_json = json.dumps(val)

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to API-friendly dictionary."""
        return {
            "id": self.id,
            "title": self.title,
            "snippet": self.article_text[:200] + ("..." if len(self.article_text) > 200 else ""),
            "input_mode": self.input_mode,
            "url_source": self.url_source,
            "domain": self.domain,
            "verdict": self.verdict,
            "confidence": self.confidence,
            "credibility_score": self.credibility_score,
            "rule_flags": self.rule_flags,
            "top_features": self.top_features,
            "created_at": self.created_at.isoformat(),
        }


class Feedback(db.Model):
    """Human-in-the-loop crowdsourced label verification model."""

    __tablename__ = "feedbacks"

    id = db.Column(db.Integer, primary_key=True)
    check_id = db.Column(db.Integer, db.ForeignKey("check_records.id", ondelete="CASCADE"), nullable=False, unique=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    user_verdict = db.Column(db.String(20), nullable=False)  # "REAL" or "FAKE"
    comment = db.Column(db.Text, nullable=True)
    status = db.Column(db.String(20), default="pending", nullable=False)  # "pending", "approved", "rejected"
    reviewed_by = db.Column(db.Integer, db.ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow, nullable=False)
    reviewed_at = db.Column(db.DateTime, nullable=True)

    def approve(self, admin_id: int):
        """Approve feedback for dataset incorporation and model retraining."""
        self.status = "approved"
        self.reviewed_by = admin_id
        self.reviewed_at = datetime.datetime.utcnow()

    def reject(self, admin_id: int):
        """Reject unverified feedback."""
        self.status = "rejected"
        self.reviewed_by = admin_id
        self.reviewed_at = datetime.datetime.utcnow()


class ActivityLog(db.Model):
    """Audit log capturing user actions, administrative interventions, and auth events."""

    __tablename__ = "activity_logs"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    action = db.Column(db.String(100), nullable=False)
    details = db.Column(db.Text, nullable=True)
    ip_address = db.Column(db.String(64), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow, nullable=False, index=True)

    @classmethod
    def log(cls, action: str, user_id: Optional[int] = None, details: Optional[str] = None, ip_address: Optional[str] = None):
        """Helper to create and persist an activity log entry."""
        log_entry = cls(
            action=action,
            user_id=user_id,
            details=details,
            ip_address=ip_address,
            created_at=datetime.datetime.utcnow(),
        )
        db.session.add(log_entry)
        db.session.commit()
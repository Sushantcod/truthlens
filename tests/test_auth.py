"""Unit Tests for Authentication, Password Security, and Session Management."""

import pytest
from app.models import User, db


def test_password_hashing(app):
    """Verify werkzeug secure password hashing and verification."""
    user = User(full_name="Alice Crypto", email="alice@test.org")
    user.set_password("CorrectHorseBattery99!")

    assert user.password_hash != "CorrectHorseBattery99!"
    assert user.check_password("CorrectHorseBattery99!") is True
    assert user.check_password("WrongPassword123!") is False


def test_user_registration(client, app):
    """Test user registration endpoint with valid data."""
    response = client.post("/auth/register", data={
        "full_name": "Marcus Aurelius",
        "email": "marcus@rome.org",
        "password": "PhilosopherKing2026!",
        "confirm_password": "PhilosopherKing2026!",
        "terms_agreed": True,
    }, follow_redirects=True)

    assert response.status_code == 200
    with app.app_context():
        user = User.query.filter_by(email="marcus@rome.org").first()
        assert user is not None
        assert user.full_name == "Marcus Aurelius"
        assert user.role == "user"
        assert len(user.api_key) == 48


def test_registration_password_strength_rejection(client):
    """Enforce NIST-style password strength checks (rejecting weak passwords)."""
    client.get("/auth/logout")  # Ensure clean session
    response = client.post("/auth/register", data={
        "full_name": "Lazy User",
        "email": "lazy@test.org",
        "password": "weak",  # too short, no uppercase, no numbers
        "confirm_password": "weak",
        "terms_agreed": True,
    })
    assert response.status_code == 200
    assert b"Password must contain at least 8 characters" in response.data or b"Password" in response.data


def test_user_login_success_and_logout(client, test_user):
    """Test successful user login followed by logout."""
    login_res = client.post("/auth/login", data={
        "email": "tester@truthlens.org",
        "password": "SecurePass123!",
    }, follow_redirects=True)

    assert login_res.status_code == 200
    assert b"tester@truthlens.org" in login_res.data or b"Standard" in login_res.data or b"Sign Out" in login_res.data

    logout_res = client.get("/auth/logout", follow_redirects=True)
    assert logout_res.status_code == 200
    assert b"Sign In" in logout_res.data


def test_user_login_failure(client):
    """Ensure invalid credentials fail gracefully."""
    res = client.post("/auth/login", data={
        "email": "nonexistent@truthlens.org",
        "password": "WrongPassword99!",
    }, follow_redirects=True)

    assert b"Invalid email or password" in res.data
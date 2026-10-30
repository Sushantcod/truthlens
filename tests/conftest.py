"""Pytest Test Configuration and Fixtures for TruthLens."""

import os
import pytest
from app import create_app, db
from app.models import User, CheckRecord


@pytest.fixture(scope="session")
def app():
    """Create test application instance using TestingConfig."""
    app_instance = create_app("testing")
    with app_instance.app_context():
        db.create_all()
        yield app_instance
        db.drop_all()


@pytest.fixture(autouse=True)
def clean_db(app):
    """Ensure database state is clean between test runs."""
    with app.app_context():
        yield
        db.session.rollback()


@pytest.fixture
def client(app):
    """A test client for dispatching HTTP requests."""
    return app.test_client()


@pytest.fixture
def runner(app):
    """A test runner for the application's Click CLI."""
    return app.test_cli_runner()


@pytest.fixture
def test_user(app):
    """Create and persist a standard test user."""
    with app.app_context():
        user = User.query.filter_by(email="tester@truthlens.org").first()
        if not user:
            user = User(full_name="Standard Tester", email="tester@truthlens.org", role="user")
            user.set_password("SecurePass123!")
            db.session.add(user)
            db.session.commit()
        # Retrieve fresh user ID and API key to prevent detached state
        db.session.refresh(user)
        return user


@pytest.fixture
def test_admin(app):
    """Create and persist an administrator test user."""
    with app.app_context():
        admin = User.query.filter_by(email="admin_test@truthlens.org").first()
        if not admin:
            admin = User(full_name="Admin Tester", email="admin_test@truthlens.org", role="admin")
            admin.set_password("AdminPass123!")
            db.session.add(admin)
            db.session.commit()
        return admin
"""TruthLens Application Configuration Module."""

import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


class Config:
    """Base application configuration."""

    SECRET_KEY = os.getenv("SECRET_KEY", "dev-truthlens-secure-default-key-replace")
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_DATABASE_URI = os.getenv(
        "DATABASE_URL", f"sqlite:///{BASE_DIR / 'truthlens.db'}"
    )

    # Flask-Limiter settings
    RATELIMIT_DEFAULT = os.getenv("RATELIMIT_DEFAULT", "200 per day;50 per hour")
    RATELIMIT_STORAGE_URI = os.getenv("RATELIMIT_STORAGE_URI", "memory://")
    RATELIMIT_STRATEGY = "fixed-window"

    # Uploads & File Limits
    MAX_CONTENT_LENGTH = 4 * 1024 * 1024  # 4 MB max upload size
    UPLOAD_FOLDER = BASE_DIR / "data" / "uploads"
    ALLOWED_EXTENSIONS = {"txt", "csv", "json"}

    # ML Model Paths
    ML_MODELS_DIR = BASE_DIR / "ml" / "models"
    MODEL_PATH = ML_MODELS_DIR / "model.pkl"
    VECTORIZER_PATH = ML_MODELS_DIR / "vectorizer.pkl"
    METRICS_PATH = ML_MODELS_DIR / "metrics.json"

    # External APIs (Optional)
    GOOGLE_FACT_CHECK_API_KEY = os.getenv("GOOGLE_FACT_CHECK_API_KEY", "").strip()
    NEWS_API_KEY = os.getenv("NEWS_API_KEY", "").strip()

    # Session & Security
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"
    WTF_CSRF_ENABLED = True
    WTF_CSRF_TIME_LIMIT = 3600

    # Human-In-The-Loop & Confidence Thresholds
    UNCERTAINTY_LOWER_BOUND = 0.40  # Below 60% probability -> Uncertain
    UNCERTAINTY_UPPER_BOUND = 0.60  # P(Fake) between 0.40 and 0.60 is Uncertain


class DevelopmentConfig(Config):
    """Development configuration."""

    DEBUG = True


class TestingConfig(Config):
    """Testing configuration with dedicated test SQLite and disabled CSRF."""

    TESTING = True
    DEBUG = False
    SQLALCHEMY_DATABASE_URI = f"sqlite:///{BASE_DIR / 'test.db'}"
    WTF_CSRF_ENABLED = False
    RATELIMIT_ENABLED = False


class ProductionConfig(Config):
    """Production configuration."""

    DEBUG = False
    SESSION_COOKIE_SECURE = True


config_by_name = {
    "development": DevelopmentConfig,
    "testing": TestingConfig,
    "production": ProductionConfig,
    "default": DevelopmentConfig,
}
"""TruthLens Application Factory and Initialization Module."""

import logging
from flask import Flask, render_template, session, request
from flask_login import LoginManager
from flask_wtf.csrf import CSRFProtect
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import click

from config import config_by_name
from app.models import db, User, CheckRecord, ActivityLog

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("truthlens.app")

login_manager = LoginManager()
csrf = CSRFProtect()
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["300 per day", "80 per hour"],
    storage_uri="memory://",
)

# Multilingual UI Dictionary (English & Hindi)
TRANSLATIONS = {
    "en": {
        "app_title": "TruthLens",
        "tagline": "AI-Powered Credibility & Fact Verification for Social Good",
        "home": "Home",
        "check": "Verify Article",
        "dashboard": "Dashboard",
        "history": "History",
        "learn": "Media Literacy",
        "admin": "Admin Studio",
        "login": "Sign In",
        "register": "Get Started",
        "logout": "Sign Out",
        "profile": "Profile",
        "hero_title": "Democratizing Truth Through Transparent AI.",
        "hero_desc": "TruthLens combines supervised natural language processing, knowledge-based reasoning, and explainable AI to help citizens, journalists, and researchers dissect misinformation before it spreads.",
        "analyze_btn": "Audit Article",
        "verdict_real": "Verified Real News",
        "verdict_fake": "High Misinformation Probability",
        "verdict_uncertain": "Uncertain / Inconclusive Evidence",
        "credibility_score": "Credibility Score",
        "confidence": "Confidence",
        "disclaimer_label": "Responsible AI Notice",
    },
    "hi": {
        "app_title": "ट्रुथ-लेंस (TruthLens)",
        "tagline": "सामाजिक भलाई के लिए एआई-संचालित समाचार सत्यापन",
        "home": "मुख्य पृष्ठ",
        "check": "समाचार जांचें",
        "dashboard": "डैशबोर्ड",
        "history": "इतिहास",
        "learn": "मीडिया साक्षरता",
        "admin": "व्यवस्थापक",
        "login": "लॉग इन",
        "register": "पंजीकरण",
        "logout": "लॉग आउट",
        "profile": "प्रोफ़ाइल",
        "hero_title": "पारदर्शी एआई के माध्यम से सत्य का सशक्तिकरण।",
        "hero_desc": "ट्रुथ-लेंस प्राकृतिक भाषा प्रसंस्करण (NLP) और व्याख्या योग्य एआई (XAI) का उपयोग करके फर्जी खबरों और भ्रामक जानकारी को उजागर करता है।",
        "analyze_btn": "सत्यापन प्रारंभ करें",
        "verdict_real": "सत्यापित वास्तविक समाचार",
        "verdict_fake": "फर्जी / भ्रामक सूचना की संभावना",
        "verdict_uncertain": "अनिश्चित / अनिर्णायक साक्ष्य",
        "credibility_score": "विश्वसनीयता स्कोर",
        "confidence": "आत्मविश्वास",
        "disclaimer_label": "जिम्मेदार एआई सूचना",
    },
}


def create_app(config_name: str = "default") -> Flask:
    """Create and configure Flask application instance."""
    app = Flask(__name__)
    app.config.from_object(config_by_name.get(config_name, config_by_name["default"]))

    # Initialize extensions
    db.init_app(app)
    login_manager.init_app(app)
    csrf.init_app(app)
    limiter.init_app(app)

    login_manager.login_view = "auth.login"
    login_manager.login_message = "Please sign in to access this feature."
    login_manager.login_message_category = "info"

    @login_manager.user_loader
    def load_user(user_id):
        return db.session.get(User, int(user_id))

    # Multilingual & UI Context Processor
    @app.context_processor
    def inject_localization():
        lang = session.get("lang", "en")
        t = TRANSLATIONS.get(lang, TRANSLATIONS["en"])
        return dict(
            current_lang=lang,
            t=t,
            year=2026,
            app_name="TruthLens",
        )

    # Register Blueprints
    from app.routes.auth import auth_bp
    from app.routes.main import main_bp
    from app.routes.admin import admin_bp
    from app.routes.api import api_bp

    csrf.exempt(api_bp)

    app.register_blueprint(auth_bp, url_prefix="/auth")
    app.register_blueprint(main_bp, url_prefix="")
    app.register_blueprint(admin_bp, url_prefix="/admin")
    app.register_blueprint(api_bp, url_prefix="/api")

    # Error Handlers
    @app.errorhandler(404)
    def page_not_found(e):
        return render_template("errors/404.html"), 404

    @app.errorhandler(500)
    def internal_server_error(e):
        return render_template("errors/500.html"), 500

    @app.errorhandler(429)
    def rate_limit_exceeded(e):
        return render_template("errors/429.html", error=str(e)), 429

    # CLI Commands
    @app.cli.command("init-db")
    def init_db():
        """Initialize SQLite database tables."""
        db.create_all()
        click.echo("Initialized database tables.")

    @app.cli.command("seed-admin")
    def seed_admin():
        """Seed initial administrator and demo user accounts."""
        db.create_all()
        admin_email = "admin@truthlens.org"
        demo_email = "user@truthlens.org"

        # Create Admin
        admin = User.query.filter_by(email=admin_email).first()
        if not admin:
            admin = User(full_name="TruthLens Director", email=admin_email, role="admin")
            admin.set_password("Admin@TruthLens2026!")
            db.session.add(admin)
            click.echo(f"Created Admin account: {admin_email} | Password: Admin@TruthLens2026!")
        else:
            admin.role = "admin"
            admin.set_password("Admin@TruthLens2026!")
            click.echo(f"Updated Admin account password: {admin_email}")

        # Create Demo User
        demo_user = User.query.filter_by(email=demo_email).first()
        if not demo_user:
            demo_user = User(full_name="Dr. Julian Vance", email=demo_email, role="user")
            demo_user.set_password("TruthSeeker2026!")
            db.session.add(demo_user)
            click.echo(f"Created Demo User: {demo_email} | Password: TruthSeeker2026!")

        db.session.commit()

        # Seed sample historical checks
        if CheckRecord.query.count() == 0:
            sample_real = CheckRecord(
                user_id=demo_user.id,
                title="Global Climate Accord Signed",
                article_text="Delegates from over 190 nations reached a landmark agreement to reduce emissions by 2030.",
                input_mode="text",
                domain="un.org",
                verdict="REAL",
                confidence=94.2,
                credibility_score=92,
                badge_color="success",
                rule_flags_json='[{"rule": "VERIFIED_REPUTABLE_SOURCE", "severity": "POSITIVE", "score_impact": 15, "description": "Domain recognized as verified institutional source."}]',
                top_features_json='[{"feature": "agreement", "weight": -0.85, "signal": "REAL", "direction": "Pushes toward REAL"}]',
            )
            sample_fake = CheckRecord(
                user_id=demo_user.id,
                title="Secret Miracle Cure Suppressed",
                article_text="SHOCKING BOMBSHELL: Big Pharma doctors hide miracle cure for all cancers!",
                input_mode="text",
                domain="naturalnews.com",
                verdict="FAKE",
                confidence=96.8,
                credibility_score=4,
                badge_color="danger",
                rule_flags_json='[{"rule": "SENSATIONAL_LEXICAL_MARKERS", "severity": "HIGH", "score_impact": -25, "description": "Contains high-frequency sensational conspiracy words."}]',
                top_features_json='[{"feature": "shocking", "weight": 1.25, "signal": "FAKE", "direction": "Pushes toward FAKE"}]',
            )
            db.session.add_all([sample_real, sample_fake])
            db.session.commit()
            click.echo("Seeded sample historical check records.")

    return app
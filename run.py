"""TruthLens Application Entrypoint.

Starts the Flask development server, ensures database schema is initialized,
and provides administrative seeding commands.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

from app import create_app, db
from app.models import User, CheckRecord

config_name = os.getenv("FLASK_ENV", "development")
app = create_app(config_name)

# Ensure database tables exist on startup
with app.app_context():
    db.create_all()

    # Auto-seed initial demo accounts if database is empty
    if User.query.count() == 0:
        admin_email = "admin@truthlens.org"
        demo_email = "user@truthlens.org"

        admin = User(full_name="TruthLens Director", email=admin_email, role="admin")
        admin.set_password("Admin@TruthLens2026!")

        demo = User(full_name="Dr. Julian Vance", email=demo_email, role="user")
        demo.set_password("TruthSeeker2026!")

        db.session.add_all([admin, demo])
        db.session.commit()

        # Seed sample audits
        sample_real = CheckRecord(
            user_id=demo.id,
            title="Global Climate Accord Finalized in Geneva",
            article_text="Delegates from over 190 nations reached a landmark agreement on Friday to reduce emissions by 2030.",
            input_mode="text",
            domain="un.org",
            verdict="REAL",
            confidence=92.5,
            credibility_score=94,
            badge_color="success",
            rule_flags_json='[{"rule": "VERIFIED_REPUTABLE_SOURCE", "severity": "POSITIVE", "score_impact": 15, "description": "Domain recognized as verified institutional source."}]',
            top_features_json='[{"feature": "agreement", "weight": -0.85, "signal": "REAL", "direction": "Pushes toward REAL"}]',
            highlighted_html='Delegates from over 190 nations reached a landmark <mark class="hl-real">agreement</mark> on Friday to reduce emissions by 2030.'
        )

        sample_fake = CheckRecord(
            user_id=demo.id,
            title="Secret Miracle Cure Suppressed by Elites",
            article_text="SHOCKING BOMBSHELL: Big Pharma doctors hide miracle cure for all cancers!",
            input_mode="text",
            domain="naturalnews.com",
            verdict="FAKE",
            confidence=97.2,
            credibility_score=6,
            badge_color="danger",
            rule_flags_json='[{"rule": "SENSATIONAL_LEXICAL_MARKERS", "severity": "HIGH", "score_impact": -25, "description": "Contains high-frequency sensational conspiracy words."}]',
            top_features_json='[{"feature": "shocking", "weight": 1.25, "signal": "FAKE", "direction": "Pushes toward FAKE"}, {"feature": "bombshell", "weight": 1.45, "signal": "FAKE", "direction": "Pushes toward FAKE"}]',
            highlighted_html='<mark class="hl-fake">SHOCKING</mark> <mark class="hl-fake">BOMBSHELL</mark>: Big Pharma doctors hide <mark class="hl-fake">miracle</mark> cure for all cancers!'
        )

        db.session.add_all([sample_real, sample_fake])
        db.session.commit()
        print("\n[TruthLens] Seeded default administrator (admin@truthlens.org) and demo auditor.\n")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "True").lower() in ("true", "1", "t")
    print(f"\n==================================================================")
    print(f"  TruthLens Server Initialized at: http://127.0.0.1:{port}")
    print(f"  Visual Style: High-Craft Editorial Newsprint (Tic-Tac-Toe Aesthetic)")
    print(f"  Admin: admin@truthlens.org | Password: Admin@TruthLens2026!")
    print(f"  Demo:  user@truthlens.org  | Password: TruthSeeker2026!")
    print(f"==================================================================\n")
    app.run(host="0.0.0.0", port=port, debug=debug)


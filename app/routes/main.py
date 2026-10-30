"""TruthLens Main Application Routes.

Implements core user workflows:
- Landing Page with Social Good Mission
- Multilingual Language Switcher (EN / HI)
- Analytical Dashboard with Chart.js telemetry
- Multi-modal News Credibility Checker (Text, URL Scraper, File Upload)
- Explainable AI Result View with Word Highlighting & Feature Attribution
- Human-in-the-Loop Feedback Submission
- Searchable, Paginated History with CSV Export
- Media Literacy Center (10-Question Interactive Quiz & Glossary)
"""

import csv
import io
import logging
from typing import Optional
from flask import (
    Blueprint,
    render_template,
    redirect,
    url_for,
    flash,
    request,
    session,
    Response,
    abort,
    send_from_directory,
    current_app,
)
from flask_login import login_required, current_user

from app.models import db, CheckRecord, Feedback, ActivityLog
from app.forms import CheckNewsForm, FeedbackForm
from app.services.scraper import scrape_article_from_url, extract_domain
from app.services.factcheck import cross_check_claim
from ml.predict import predict

logger = logging.getLogger("truthlens.main")
main_bp = Blueprint("main", __name__)


@main_bp.route("/favicon.ico")
def favicon():
    """Serve favicon directly from root path."""
    import os
    return send_from_directory(
        os.path.join(current_app.root_path, "static"),
        "favicon.ico",
        mimetype="image/vnd.microsoft.icon",
    )


@main_bp.route("/")
def index():
    """Landing page showcasing the TruthLens mission, architecture, and impact."""
    total_audits = CheckRecord.query.count()
    return render_template("main/index.html", total_audits=total_audits)


@main_bp.route("/set-language/<lang>")
def set_language(lang):
    """Toggle language preference between English ('en') and Hindi ('hi')."""
    if lang in ["en", "hi"]:
        session["lang"] = lang
    next_page = request.referrer or url_for("main.index")
    return redirect(next_page)


@main_bp.route("/dashboard")
@login_required
def dashboard():
    """Personal analytics dashboard displaying audit metrics and trend lines."""
    user_checks = CheckRecord.query.filter_by(user_id=current_user.id).order_by(CheckRecord.created_at.desc())
    total_checks = user_checks.count()

    fake_count = user_checks.filter_by(verdict="FAKE").count()
    real_count = user_checks.filter_by(verdict="REAL").count()
    uncertain_count = user_checks.filter_by(verdict="UNCERTAIN").count()

    fake_pct = round((fake_count / total_checks) * 100, 1) if total_checks > 0 else 0
    real_pct = round((real_count / total_checks) * 100, 1) if total_checks > 0 else 0

    # Recent 6 audits
    recent_records = user_checks.limit(6).all()

    # Aggregate scores for trend chart
    recent_trend = [
        {"date": r.created_at.strftime("%b %d"), "score": r.credibility_score}
        for r in reversed(user_checks.limit(10).all())
    ]

    return render_template(
        "main/dashboard.html",
        total_checks=total_checks,
        fake_count=fake_count,
        real_count=real_count,
        uncertain_count=uncertain_count,
        fake_pct=fake_pct,
        real_pct=real_pct,
        recent_records=recent_records,
        recent_trend=recent_trend,
    )


@main_bp.route("/check", methods=["GET", "POST"])
@login_required
def check_news():
    """Multi-modal article verification entrypoint."""
    form = CheckNewsForm()

    if form.validate_on_submit():
        mode = form.input_mode.data
        title = form.title.data.strip() if form.title.data else ""
        text_to_audit = ""
        domain = ""
        url_source = ""

        if mode == "text":
            text_to_audit = (form.article_text.data or "").strip()
            if not text_to_audit:
                flash("Please input an article excerpt, statement, or headline to analyze.", "warning")
                return render_template("main/check.html", form=form)

        elif mode == "url":
            url_source = (form.article_url.data or "").strip()
            if not url_source:
                flash("Please enter a valid news URL.", "warning")
                return render_template("main/check.html", form=form)

            scrape_res = scrape_article_from_url(url_source)
            if not scrape_res["success"]:
                flash(f"Scraper notice: {scrape_res['error']}", "danger")
                return render_template("main/check.html", form=form)

            text_to_audit = scrape_res["text"]
            domain = scrape_res["domain"]
            if not title:
                title = scrape_res.get("title", "")

        elif mode == "file":
            file = form.article_file.data
            if not file:
                flash("Please choose a plain text (.txt) file to upload.", "warning")
                return render_template("main/check.html", form=form)

            try:
                text_to_audit = file.read().decode("utf-8", errors="replace").strip()
                if not title:
                    title = file.filename
            except Exception as e:
                flash(f"Failed to read file: {str(e)}", "danger")
                return render_template("main/check.html", form=form)

        if len(text_to_audit.split()) < 3:
            flash("The provided text is too brief to analyze. Please submit at least a full sentence.", "warning")
            return render_template("main/check.html", form=form)

        # Execute ML inference & Knowledge Rule evaluation
        prediction_output = predict(text_to_audit, domain=domain)

        # Persist audit record in database
        check_record = CheckRecord(
            user_id=current_user.id,
            title=title or text_to_audit[:60] + "...",
            article_text=text_to_audit,
            input_mode=mode,
            url_source=url_source,
            domain=domain,
            verdict=prediction_output["verdict"],
            confidence=prediction_output["confidence"],
            credibility_score=prediction_output["credibility_score"],
            badge_color=prediction_output["badge_color"],
            rule_flags_json=prediction_output.get("rule_flags"),
            top_features_json=prediction_output.get("top_features"),
            highlighted_html=prediction_output.get("highlighted_html"),
        )
        # Store JSON properties via setter
        check_record.rule_flags = prediction_output.get("rule_flags", [])
        check_record.top_features = prediction_output.get("top_features", [])

        db.session.add(check_record)
        db.session.commit()

        ActivityLog.log(
            action="ARTICLE_AUDITED",
            user_id=current_user.id,
            details=f"Audited claim: {check_record.title[:40]} | Verdict: {check_record.verdict}",
            ip_address=request.remote_addr,
        )

        return redirect(url_for("main.result", check_id=check_record.id))

    return render_template("main/check.html", form=form)


@main_bp.route("/result/<int:check_id>", methods=["GET"])
@login_required
def result(check_id):
    """Present comprehensive explainable AI findings and external fact-checks."""
    record = db.session.get(CheckRecord, check_id)
    if not record or (record.user_id != current_user.id and not current_user.is_admin):
        abort(404)

    feedback_form = FeedbackForm()
    user_feedback = Feedback.query.filter_by(check_id=record.id, user_id=current_user.id).first()

    # Optional cross-check against external fact checks
    search_query = record.title if record.title else record.article_text[:120]
    factcheck_results = cross_check_claim(search_query)

    return render_template(
        "main/result.html",
        record=record,
        feedback_form=feedback_form,
        user_feedback=user_feedback,
        factcheck_results=factcheck_results,
    )


@main_bp.route("/feedback/<int:check_id>", methods=["POST"])
@login_required
def submit_feedback(check_id):
    """Record crowdsourced user audit feedback for human-in-the-loop validation."""
    record = db.session.get(CheckRecord, check_id)
    if not record:
        abort(404)

    form = FeedbackForm()
    if form.validate_on_submit():
        existing = Feedback.query.filter_by(check_id=record.id, user_id=current_user.id).first()
        if existing:
            existing.user_verdict = form.user_verdict.data
            existing.comment = form.comment.data
            flash("Your previous feedback evaluation has been updated.", "info")
        else:
            fb = Feedback(
                check_id=record.id,
                user_id=current_user.id,
                user_verdict=form.user_verdict.data,
                comment=form.comment.data,
            )
            db.session.add(fb)
            flash("Thank you for contributing to our human-in-the-loop verification dataset!", "success")

        db.session.commit()
        ActivityLog.log(
            action="FEEDBACK_SUBMITTED",
            user_id=current_user.id,
            details=f"Feedback submitted for Check #{record.id}: {form.user_verdict.data}",
            ip_address=request.remote_addr,
        )

    return redirect(url_for("main.result", check_id=check_id))


@main_bp.route("/history")
@login_required
def history():
    """Searchable, filterable, paginated archive of past verifications."""
    page = request.args.get("page", 1, type=int)
    search = request.args.get("q", "").strip()
    verdict_filter = request.args.get("verdict", "").strip()

    query = CheckRecord.query.filter_by(user_id=current_user.id)

    if search:
        query = query.filter(
            (CheckRecord.title.ilike(f"%{search}%")) | (CheckRecord.article_text.ilike(f"%{search}%"))
        )
    if verdict_filter in ["REAL", "FAKE", "UNCERTAIN"]:
        query = query.filter_by(verdict=verdict_filter)

    pagination = query.order_by(CheckRecord.created_at.desc()).paginate(page=page, per_page=8, error_out=False)

    return render_template(
        "main/history.html",
        pagination=pagination,
        records=pagination.items,
        search=search,
        verdict_filter=verdict_filter,
    )


@main_bp.route("/history/delete/<int:check_id>", methods=["POST"])
@login_required
def delete_history_item(check_id):
    """Delete a single verification record."""
    record = db.session.get(CheckRecord, check_id)
    if not record or (record.user_id != current_user.id and not current_user.is_admin):
        abort(404)

    db.session.delete(record)
    db.session.commit()
    flash("Verification record deleted.", "info")
    return redirect(url_for("main.history"))


@main_bp.route("/history/export")
@login_required
def export_history_csv():
    """Export past checks to CSV format."""
    records = CheckRecord.query.filter_by(user_id=current_user.id).order_by(CheckRecord.created_at.desc()).all()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "ID", "Date", "Title", "Input Mode", "Domain", "Verdict",
        "Confidence (%)", "Credibility Score (0-100)", "Snippet"
    ])

    for r in records:
        writer.writerow([
            r.id,
            r.created_at.strftime("%Y-%m-%d %H:%M"),
            r.title,
            r.input_mode,
            r.domain or "N/A",
            r.verdict,
            r.confidence,
            r.credibility_score,
            r.article_text[:150].replace("\n", " "),
        ])

    output.seek(0)
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=truthlens_history.csv"},
    )


@main_bp.route("/learn")
def learn():
    """Media Literacy Center featuring tips, glossary, and a 10-question quiz."""
    return render_template("main/learn.html")


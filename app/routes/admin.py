"""TruthLens Administrator Studio & Model Governance Routes.

Implements:
- Admin Telemetry Dashboard (User metrics, Daily audit trends, Feedback volumes)
- User Management (View, Deactivate/Reactivate, Promote to Admin, Delete)
- Human-in-the-Loop Feedback Review Queue (Approve / Reject crowd labels)
- Automated Model Retraining with Feedback Loop & Version Rollback
- Model Performance Telemetry (Live Confusion Matrix, ROC curves, Metrics comparison)
- System Audit Logs
"""

import datetime
import json
import logging
import shutil
from pathlib import Path
from flask import (
    Blueprint,
    render_template,
    redirect,
    url_for,
    flash,
    request,
    abort,
)
from flask_login import login_required, current_user
import pandas as pd

from app.models import db, User, CheckRecord, Feedback, ActivityLog
from ml.preprocess import preprocess_for_tfidf

logger = logging.getLogger("truthlens.admin")
admin_bp = Blueprint("admin", __name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = BASE_DIR / "ml" / "models"
DATA_PROCESSED_PATH = BASE_DIR / "data" / "processed" / "news_dataset.csv"


def admin_required(f):
    """Decorator ensuring request originates from an authenticated administrator."""
    from functools import wraps

    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or not current_user.is_admin:
            flash("Administrator clearance required to access this resource.", "danger")
            return redirect(url_for("main.index"))
        return f(*args, **kwargs)

    return decorated_function


@admin_bp.route("/")
@login_required
@admin_required
def dashboard():
    """Administrative command center overview."""
    total_users = User.query.count()
    total_checks = CheckRecord.query.count()
    pending_feedback_count = Feedback.query.filter_by(status="pending").count()
    approved_feedback_count = Feedback.query.filter_by(status="approved").count()

    # Load active model metrics
    metrics_path = MODELS_DIR / "metrics.json"
    metrics_data = {}
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics_data = json.load(f)

    # Activity in past 7 days
    today = datetime.datetime.utcnow().date()
    days = [(today - datetime.timedelta(days=i)) for i in range(6, -1, -1)]
    checks_by_day = []
    for d in days:
        cnt = CheckRecord.query.filter(
            db.func.date(CheckRecord.created_at) == d
        ).count()
        checks_by_day.append({"date": d.strftime("%b %d"), "count": cnt})

    recent_logs = ActivityLog.query.order_by(ActivityLog.created_at.desc()).limit(10).all()

    return render_template(
        "admin/dashboard.html",
        total_users=total_users,
        total_checks=total_checks,
        pending_feedback_count=pending_feedback_count,
        approved_feedback_count=approved_feedback_count,
        metrics=metrics_data,
        checks_by_day=checks_by_day,
        recent_logs=recent_logs,
    )


@admin_bp.route("/users")
@login_required
@admin_required
def users():
    """User directory and privilege governance."""
    all_users = User.query.order_by(User.created_at.desc()).all()
    return render_template("admin/users.html", users=all_users)


@admin_bp.route("/users/<int:user_id>/toggle-status", methods=["POST"])
@login_required
@admin_required
def toggle_user_status(user_id):
    """Deactivate or reactivate a user account."""
    if user_id == current_user.id:
        flash("You cannot deactivate your own active session.", "warning")
        return redirect(url_for("admin.users"))

    user = db.session.get(User, user_id)
    if not user:
        abort(404)

    user.is_active_account = not user.is_active_account
    db.session.commit()

    status_str = "activated" if user.is_active_account else "deactivated"
    ActivityLog.log(
        action=f"USER_{status_str.upper()}",
        user_id=current_user.id,
        details=f"Admin {current_user.email} {status_str} account {user.email}",
        ip_address=request.remote_addr,
    )
    flash(f"Account for {user.email} has been {status_str}.", "info")
    return redirect(url_for("admin.users"))


@admin_bp.route("/users/<int:user_id>/toggle-role", methods=["POST"])
@login_required
@admin_required
def toggle_user_role(user_id):
    """Promote user to administrator or demote back to standard user."""
    if user_id == current_user.id:
        flash("You cannot revoke your own administrative status.", "warning")
        return redirect(url_for("admin.users"))

    user = db.session.get(User, user_id)
    if not user:
        abort(404)

    user.role = "user" if user.role == "admin" else "admin"
    db.session.commit()

    ActivityLog.log(
        action="USER_ROLE_CHANGED",
        user_id=current_user.id,
        details=f"Admin {current_user.email} changed role of {user.email} to {user.role}",
        ip_address=request.remote_addr,
    )
    flash(f"Role for {user.email} updated to '{user.role}'.", "success")
    return redirect(url_for("admin.users"))


@admin_bp.route("/users/<int:user_id>/delete", methods=["POST"])
@login_required
@admin_required
def delete_user(user_id):
    """Permanently delete a user account and associated history."""
    if user_id == current_user.id:
        flash("You cannot delete your own active administrator account.", "warning")
        return redirect(url_for("admin.users"))

    user = db.session.get(User, user_id)
    if not user:
        abort(404)

    deleted_email = user.email
    db.session.delete(user)
    db.session.commit()

    ActivityLog.log(
        action="USER_DELETED_BY_ADMIN",
        user_id=current_user.id,
        details=f"Admin {current_user.email} deleted account {deleted_email}",
        ip_address=request.remote_addr,
    )
    flash(f"User {deleted_email} deleted successfully.", "info")
    return redirect(url_for("admin.users"))


@admin_bp.route("/feedback")
@login_required
@admin_required
def feedback_queue():
    """Review crowdsourced human-in-the-loop labels."""
    status_filter = request.args.get("status", "pending")
    feedbacks = Feedback.query.filter_by(status=status_filter).order_by(Feedback.created_at.desc()).all()
    return render_template("admin/feedback.html", feedbacks=feedbacks, active_status=status_filter)


@admin_bp.route("/feedback/<int:fb_id>/<action>", methods=["POST"])
@login_required
@admin_required
def handle_feedback(fb_id, action):
    """Approve or reject a submitted feedback label."""
    fb = db.session.get(Feedback, fb_id)
    if not fb:
        abort(404)

    if action == "approve":
        fb.approve(current_user.id)
        flash("Feedback label approved for incorporation into next retraining cycle.", "success")
    elif action == "reject":
        fb.reject(current_user.id)
        flash("Feedback label rejected.", "info")

    db.session.commit()
    return redirect(url_for("admin.feedback_queue", status="pending"))


@admin_bp.route("/retrain", methods=["POST"])
@login_required
@admin_required
def retrain_model():
    """Trigger retraining on original dataset plus approved human feedback."""
    approved_feedback = Feedback.query.filter_by(status="approved").all()
    new_rows = []

    for fb in approved_feedback:
        record = fb.check_record
        if record and record.article_text:
            label_val = 1 if fb.user_verdict == "FAKE" else 0
            label_name = "FAKE" if label_val == 1 else "REAL"
            new_rows.append({
                "title": record.title or "User Verified Sample",
                "text": record.article_text,
                "subject": "human_feedback",
                "label": label_val,
                "label_name": label_name,
            })

    # Backup current model before retraining to enable rollback
    prod_model = MODELS_DIR / "model.pkl"
    backup_model = MODELS_DIR / "model_previous_backup.pkl"
    if prod_model.exists():
        shutil.copy2(prod_model, backup_model)

    # Append new feedback data to training dataset if any exist
    if new_rows and DATA_PROCESSED_PATH.exists():
        df_existing = pd.read_csv(DATA_PROCESSED_PATH)
        df_new = pd.DataFrame(new_rows)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.to_csv(DATA_PROCESSED_PATH, index=False)
        logger.info("Integrated %d approved feedback samples into dataset.", len(new_rows))

    # Invoke reproducible training pipeline
    from ml.train import train_models_pipeline
    metrics = train_models_pipeline()

    ActivityLog.log(
        action="MODEL_RETRAINED",
        user_id=current_user.id,
        details=f"Retrained on {metrics['total_samples']} samples (including {len(new_rows)} feedback entries).",
        ip_address=request.remote_addr,
    )

    flash(f"Model retrained successfully on {metrics['total_samples']} records! New accuracy: {metrics['models']['Soft-Voting Ensemble']['accuracy']*100:.1f}%.", "success")
    return redirect(url_for("admin.performance"))


@admin_bp.route("/rollback", methods=["POST"])
@login_required
@admin_required
def rollback_model():
    """Roll back to the previous model checkpoint."""
    prod_model = MODELS_DIR / "model.pkl"
    backup_model = MODELS_DIR / "model_previous_backup.pkl"

    if not backup_model.exists():
        flash("No previous model backup available for rollback.", "warning")
        return redirect(url_for("admin.performance"))

    shutil.copy2(backup_model, prod_model)
    ActivityLog.log(
        action="MODEL_ROLLED_BACK",
        user_id=current_user.id,
        details="Restored previous model checkpoint",
        ip_address=request.remote_addr,
    )
    flash("Model successfully restored to previous version checkpoint.", "info")
    return redirect(url_for("admin.performance"))


@admin_bp.route("/performance")
@login_required
@admin_required
def performance():
    """Display comprehensive model evaluation metrics, plots, and error analysis."""
    metrics_path = MODELS_DIR / "metrics.json"
    metrics_data = {}
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics_data = json.load(f)

    return render_template("admin/performance.html", metrics=metrics_data)


@admin_bp.route("/logs")
@login_required
@admin_required
def activity_logs():
    """System activity audit trail."""
    page = request.args.get("page", 1, type=int)
    pagination = ActivityLog.query.order_by(ActivityLog.created_at.desc()).paginate(page=page, per_page=20, error_out=False)
    return render_template("admin/logs.html", pagination=pagination, logs=pagination.items)


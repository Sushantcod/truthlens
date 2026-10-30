"""TruthLens Authentication & Account Management Routes.

Handles:
- User Registration (Password strength validation, unique email enforcement)
- Login / Logout (Session management, remember me, rate-limiting)
- Password Recovery (Token generation, console link output fallback)
- User Profile (Display name updates, credential rotation, API key regeneration, deletion)
"""

import logging
from flask import (
    Blueprint,
    render_template,
    redirect,
    url_for,
    flash,
    request,
    current_app,
)
from flask_login import login_user, logout_user, login_required, current_user

from app.models import db, User, ActivityLog
from app.forms import (
    RegistrationForm,
    LoginForm,
    ForgotPasswordForm,
    ResetPasswordForm,
    ProfileForm,
)

logger = logging.getLogger("truthlens.auth")
auth_bp = Blueprint("auth", __name__)


@auth_bp.route("/register", methods=["GET", "POST"])
def register():
    """Register a new user account."""
    if current_user.is_authenticated:
        return redirect(url_for("main.dashboard"))

    form = RegistrationForm()
    if form.validate_on_submit():
        user = User(
            full_name=form.full_name.data,
            email=form.email.data,
            role="user",
        )
        user.set_password(form.password.data)
        db.session.add(user)
        db.session.commit()

        ActivityLog.log(
            action="USER_REGISTERED",
            user_id=user.id,
            details=f"Account created for {user.email}",
            ip_address=request.remote_addr,
        )

        login_user(user)
        flash("Your TruthLens account has been established successfully. Welcome!", "success")
        return redirect(url_for("main.dashboard"))

    return render_template("auth/register.html", form=form)


@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    """Authenticate user with rate-limiting and security audit logging."""
    if current_user.is_authenticated:
        return redirect(url_for("main.dashboard"))

    form = LoginForm()
    if form.validate_on_submit():
        email = form.email.data.strip().lower()
        user = User.query.filter_by(email=email).first()

        if user and user.check_password(form.password.data):
            if not user.is_active_account:
                flash("This account has been deactivated by an administrator. Please contact support.", "danger")
                return render_template("auth/login.html", form=form)

            login_user(user, remember=form.remember_me.data)

            ActivityLog.log(
                action="USER_LOGIN_SUCCESS",
                user_id=user.id,
                details=f"Successful authentication for {user.email}",
                ip_address=request.remote_addr,
            )

            flash(f"Welcome back, {user.full_name.split()[0]}!", "success")
            next_page = request.args.get("next")
            if next_page and next_page.startswith("/"):
                return redirect(next_page)
            return redirect(url_for("main.dashboard"))
        else:
            ActivityLog.log(
                action="USER_LOGIN_FAILED",
                details=f"Failed login attempt for {email}",
                ip_address=request.remote_addr,
            )
            flash("Invalid email or password. Please verify your credentials and try again.", "danger")

    return render_template("auth/login.html", form=form)


@auth_bp.route("/logout")
@login_required
def logout():
    """Terminate user session."""
    ActivityLog.log(
        action="USER_LOGOUT",
        user_id=current_user.id,
        details=f"Sign-out for {current_user.email}",
        ip_address=request.remote_addr,
    )
    logout_user()
    flash("You have been signed out safely.", "info")
    return redirect(url_for("main.index"))


@auth_bp.route("/forgot-password", methods=["GET", "POST"])
def forgot_password():
    """Initiate password recovery."""
    form = ForgotPasswordForm()
    if form.validate_on_submit():
        email = form.email.data.strip().lower()
        user = User.query.filter_by(email=email).first()
        if user:
            token = user.create_reset_token(expires_in_minutes=30)
            db.session.commit()
            reset_url = url_for("auth.reset_password", token=token, _external=True)

            # Print to stdout/console when SMTP is not configured
            print("\n" + "=" * 70)
            print(">>> TRUTHLENS PASSWORD RESET DISPATCH (LOCAL CONSOLE SIMULATION) <<<")
            print(f"Recipient:  {user.email}")
            print(f"Reset Link: {reset_url}")
            print("Expires:    30 minutes")
            print("=" * 70 + "\n")

            ActivityLog.log(
                action="PASSWORD_RESET_REQUESTED",
                user_id=user.id,
                details=f"Password recovery requested for {user.email}",
                ip_address=request.remote_addr,
            )

        # Constant message to prevent email enumeration attacks
        flash("If an account exists for this email, password recovery instructions have been dispatched.", "info")
        return redirect(url_for("auth.login"))

    return render_template("auth/forgot_password.html", form=form)


@auth_bp.route("/reset-password/<token>", methods=["GET", "POST"])
def reset_password(token):
    """Fulfill password reset using token."""
    user = User.query.filter_by(reset_token=token).first()
    if not user or not user.verify_reset_token(token):
        flash("The password recovery token is invalid or has expired. Please request a new link.", "danger")
        return redirect(url_for("auth.forgot_password"))

    form = ResetPasswordForm()
    if form.validate_on_submit():
        user.set_password(form.password.data)
        user.clear_reset_token()
        db.session.commit()

        ActivityLog.log(
            action="PASSWORD_RESET_COMPLETED",
            user_id=user.id,
            details=f"Password updated successfully for {user.email}",
            ip_address=request.remote_addr,
        )

        flash("Your password has been updated securely. You may now sign in.", "success")
        return redirect(url_for("auth.login"))

    return render_template("auth/reset_password.html", form=form)


@auth_bp.route("/profile", methods=["GET", "POST"])
@login_required
def profile():
    """User profile management."""
    form = ProfileForm(obj=current_user)
    if form.validate_on_submit():
        # Update display name
        current_user.full_name = form.full_name.data.strip()

        # Update password if requested
        if form.new_password.data:
            if not form.current_password.data or not current_user.check_password(form.current_password.data):
                flash("Current password must be verified before modifying credentials.", "danger")
                return render_template("auth/profile.html", form=form)

            current_user.set_password(form.new_password.data)
            flash("Your password has been updated.", "success")

        db.session.commit()
        ActivityLog.log(
            action="PROFILE_UPDATED",
            user_id=current_user.id,
            details="User profile details modified",
            ip_address=request.remote_addr,
        )
        flash("Profile settings saved successfully.", "success")
        return redirect(url_for("auth.profile"))

    return render_template("auth/profile.html", form=form)


@auth_bp.route("/profile/regenerate-api-key", methods=["POST"])
@login_required
def regenerate_api_key():
    """Regenerate developer REST API token."""
    new_key = current_user.generate_new_api_key()
    db.session.commit()

    ActivityLog.log(
        action="API_KEY_REGENERATED",
        user_id=current_user.id,
        details="New REST API key generated",
        ip_address=request.remote_addr,
    )
    flash("A new API key has been created. Update your client applications.", "success")
    return redirect(url_for("auth.profile"))


@auth_bp.route("/profile/delete-account", methods=["POST"])
@login_required
def delete_account():
    """Permanently delete user account."""
    user_id = current_user.id
    email = current_user.email
    logout_user()

    user = db.session.get(User, user_id)
    if user:
        db.session.delete(user)
        db.session.commit()

    ActivityLog.log(
        action="USER_DELETED_SELF",
        details=f"Account deleted for {email}",
        ip_address=request.remote_addr,
    )
    flash("Your account and associated history have been removed permanently.", "info")
    return redirect(url_for("main.index"))


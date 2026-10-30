"""TruthLens Secure Web Forms Module.

Implements CSRF-protected, validated forms using WTForms:
- User Registration (Password strength, format validation, duplicate checks)
- User Authentication (Login, Remember Me, Rate-limiting compliance)
- Password Recovery (Forgot / Reset Token forms)
- Profile Management (Name updates, Password rotation, Account deletion)
- News Verification Input (Multimodal: Direct Text, URL Scrape, File Upload)
- Human-in-the-Loop Feedback (User assertion, citations, comments)
"""

import re
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed
from wtforms import (
    StringField,
    PasswordField,
    BooleanField,
    SubmitField,
    TextAreaField,
    RadioField,
    HiddenField,
)
from wtforms.validators import (
    DataRequired,
    Email,
    EqualTo,
    Length,
    ValidationError,
    Optional as OptionalValidator,
    URL,
)
from app.models import User


def validate_password_strength(form, field):
    """Enforce NIST-aligned strong password hygiene: >=8 chars, digits, mixed case."""
    password = field.data
    if len(password) < 8:
        raise ValidationError("Password must contain at least 8 characters.")
    if not re.search(r"\d", password):
        raise ValidationError("Password must contain at least one numerical digit.")
    if not re.search(r"[A-Z]", password):
        raise ValidationError("Password must contain at least one uppercase letter.")
    if not re.search(r"[a-z]", password):
        raise ValidationError("Password must contain at least one lowercase letter.")


class RegistrationForm(FlaskForm):
    """User account creation form with validation."""

    full_name = StringField(
        "Full Name",
        validators=[DataRequired(message="Name is required."), Length(min=2, max=100)],
        render_kw={"placeholder": "e.g. Elena Rostova", "autofocus": True},
    )
    email = StringField(
        "Email Address",
        validators=[DataRequired(), Email(message="Please provide a valid email address.")],
        render_kw={"placeholder": "name@organization.com"},
    )
    password = PasswordField(
        "Password",
        validators=[
            DataRequired(),
            validate_password_strength,
        ],
        render_kw={"placeholder": "Minimum 8 chars, mixed case and digits", "id": "reg_password"},
    )
    confirm_password = PasswordField(
        "Confirm Password",
        validators=[
            DataRequired(),
            EqualTo("password", message="Passwords must match exactly."),
        ],
        render_kw={"placeholder": "Repeat password to confirm"},
    )
    terms_agreed = BooleanField(
        "I acknowledge that TruthLens assists human fact-checking responsibly",
        validators=[DataRequired(message="You must agree to the responsible usage terms.")],
    )
    submit = SubmitField("Create Free Account")

    def validate_email(self, field):
        """Ensure email uniqueness."""
        existing = User.query.filter_by(email=field.data.strip().lower()).first()
        if existing:
            raise ValidationError("An account with this email address already exists.")


class LoginForm(FlaskForm):
    """User sign-in form with remember me functionality."""

    email = StringField(
        "Email Address",
        validators=[DataRequired(), Email(message="Please enter a valid email.")],
        render_kw={"placeholder": "you@organization.com", "autofocus": True},
    )
    password = PasswordField(
        "Password",
        validators=[DataRequired(message="Password is required.")],
        render_kw={"placeholder": "Your confidential password", "id": "login_password"},
    )
    remember_me = BooleanField("Keep me signed in on this workstation")
    submit = SubmitField("Sign In")


class ForgotPasswordForm(FlaskForm):
    """Password recovery request form."""

    email = StringField(
        "Account Email",
        validators=[DataRequired(), Email()],
        render_kw={"placeholder": "Registered email address", "autofocus": True},
    )
    submit = SubmitField("Send Recovery Instructions")


class ResetPasswordForm(FlaskForm):
    """Password reset form using verification token."""

    password = PasswordField(
        "New Password",
        validators=[DataRequired(), validate_password_strength],
        render_kw={"placeholder": "New secure password", "id": "reset_password"},
    )
    confirm_password = PasswordField(
        "Confirm New Password",
        validators=[DataRequired(), EqualTo("password", message="Passwords must match.")],
        render_kw={"placeholder": "Confirm new password"},
    )
    submit = SubmitField("Update Password")


class ProfileForm(FlaskForm):
    """User profile update form."""

    full_name = StringField(
        "Display Name",
        validators=[DataRequired(), Length(min=2, max=100)],
    )
    current_password = PasswordField(
        "Current Password (leave blank if keeping unchanged)",
        validators=[OptionalValidator()],
    )
    new_password = PasswordField(
        "New Password (optional)",
        validators=[OptionalValidator(), validate_password_strength],
    )
    confirm_new_password = PasswordField(
        "Confirm New Password",
        validators=[
            OptionalValidator(),
            EqualTo("new_password", message="New passwords must match."),
        ],
    )
    submit = SubmitField("Save Profile Changes")


class CheckNewsForm(FlaskForm):
    """Multimodal news verification submission form."""

    input_mode = RadioField(
        "Input Mode",
        choices=[("text", "Paste Text / Headline"), ("url", "Article URL Scraper"), ("file", "Upload .txt File")],
        default="text",
    )
    title = StringField(
        "Headline / Claim Title (Optional)",
        validators=[OptionalValidator(), Length(max=255)],
        render_kw={"placeholder": "Optional headline or claim summary..."},
    )
    article_text = TextAreaField(
        "Article Text or Statement",
        validators=[OptionalValidator()],
        render_kw={"rows": 6, "placeholder": "Paste news article excerpt, press release, or viral social post..."},
    )
    article_url = StringField(
        "News Article URL",
        validators=[OptionalValidator(), URL(message="Please provide a valid HTTP/HTTPS URL.")],
        render_kw={"placeholder": "https://example.com/news/article-headline"},
    )
    article_file = FileField(
        "Text Document (.txt)",
        validators=[OptionalValidator(), FileAllowed(["txt"], "Only plain text (.txt) files are accepted.")],
    )
    submit = SubmitField("Analyze Credibility")


class FeedbackForm(FlaskForm):
    """Human-in-the-loop audit feedback submission form."""

    user_verdict = RadioField(
        "Your Assessment",
        choices=[("REAL", "Verified Real News"), ("FAKE", "Fabricated / Disinformation")],
        validators=[DataRequired(message="Please select your assessment.")],
    )
    comment = TextAreaField(
        "Rationale & Evidence Link (Optional)",
        validators=[OptionalValidator(), Length(max=1000)],
        render_kw={"rows": 3, "placeholder": "Share factual context, official primary source link, or correction..."},
    )
    submit = SubmitField("Submit Audit Feedback")


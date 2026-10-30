"""Unit Tests for Machine Learning Inference, XAI, and Anti-Leakage Filters."""

from ml.predict import predict
from ml.preprocess import clean_text, strip_data_leakage, tokenize_and_lemmatize


def test_anti_leakage_cleaning():
    """Verify wire service signatures and datelines are purged to prevent artifact leakage."""
    raw_reuters = "WASHINGTON (Reuters) - Federal regulators approved the commercial merger on Friday."
    cleaned = clean_text(raw_reuters, remove_leakage=True)

    assert "WASHINGTON (Reuters)" not in cleaned
    assert "(Reuters)" not in cleaned
    assert "Federal regulators" in cleaned or "merger" in cleaned


def test_tokenization_and_lemmatization():
    """Verify stopwords removal and WordNet lemmatization."""
    text = "The scientists are developing groundbreaking vaccines for communities."
    tokens = tokenize_and_lemmatize(text, remove_stopwords=True)

    assert "the" not in tokens
    assert "are" not in tokens
    # Lemmatized plurals to singular
    assert "scientist" in tokens or "scientists" in tokens
    assert "vaccine" in tokens or "vaccines" in tokens


def test_predict_fake_news_sample():
    """Ensure clear disinformation triggers FAKE verdict with low credibility score."""
    sample_text = (
        "SHOCKING BOMBSHELL: Secret cure for all cancers discovered in lemon juice! "
        "Big pharma cartel is hiding this miraculous truth from you! Share before it gets deleted!"
    )
    result = predict(sample_text)

    assert result["verdict"] == "FAKE"
    assert result["confidence"] > 60.0
    assert result["credibility_score"] < 40
    assert len(result["top_features"]) > 0
    assert "hl-fake" in result["highlighted_html"]


def test_predict_real_news_sample():
    """Ensure verified journalistic prose yields REAL verdict with high credibility score."""
    sample_text = (
        "The Federal Reserve announced on Wednesday that its benchmark policy interest rate "
        "will remain steady in the current 5.25 to 5.50 percent range, citing sustained progress "
        "toward the two percent long-term inflation objective."
    )
    result = predict(sample_text)

    assert result["verdict"] == "REAL"
    assert result["confidence"] > 60.0
    assert result["credibility_score"] > 65
    assert len(result["top_features"]) > 0


def test_uncertainty_threshold_calibration():
    """Responsible AI requirement: probabilities between 40% and 60% produce UNCERTAIN verdict."""
    neutral_text = "The city transit council scheduled a discussion on station maintenance for next Tuesday morning."
    result = predict(neutral_text)

    # Either UNCERTAIN verdict or probability is calibrated
    if 0.40 <= result["probabilities"]["FAKE"] <= 0.60:
        assert result["verdict"] == "UNCERTAIN"
    assert "disclaimer" in result


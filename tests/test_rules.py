"""Unit Tests for Knowledge-Based Rules and Heuristic Credibility Evaluation."""

from ml.predict import check_knowledge_rules, calculate_credibility_score


def test_blacklisted_domain_rule():
    """Verify known disinformation domain watchlist triggers high-severity flag."""
    text = "A simple update regarding regional events."
    flags = check_knowledge_rules(text, domain="infowars.com")

    rule_names = [f["rule"] for f in flags]
    assert "BLACKLISTED_UNRELIABLE_SOURCE" in rule_names
    # Check that score impact is negative
    domain_flag = next(f for f in flags if f["rule"] == "BLACKLISTED_UNRELIABLE_SOURCE")
    assert domain_flag["score_impact"] <= -30


def test_reputable_domain_rule():
    """Verify reputable wire domain triggers positive credibility bonus."""
    text = "A simple report on international diplomatic developments."
    flags = check_knowledge_rules(text, domain="reuters.com")

    rule_names = [f["rule"] for f in flags]
    assert "VERIFIED_REPUTABLE_SOURCE" in rule_names
    domain_flag = next(f for f in flags if f["rule"] == "VERIFIED_REPUTABLE_SOURCE")
    assert domain_flag["score_impact"] > 0


def test_excessive_capitalization_rule():
    """Verify high uppercase ratio triggers capitalization penalty."""
    text = "THIS IS A COMPLETELY FABRICATED STATEMENT DESIGNED TO ALARM CITIZENS EVERYWHERE!"
    flags = check_knowledge_rules(text)

    rule_names = [f["rule"] for f in flags]
    assert "EXCESSIVE_CAPITALIZATION" in rule_names


def test_sensational_lexicon_rule():
    """Verify presence of sensational clickbait phrases triggers penalty."""
    text = "A shocking bombshell leak reveals secret hidden truth that they don't want you to know!"
    flags = check_knowledge_rules(text)

    rule_names = [f["rule"] for f in flags]
    assert "SENSATIONAL_LEXICAL_MARKERS" in rule_names


def test_credibility_score_clipping():
    """Verify calculated score is strictly bound within [0, 100]."""
    # Extreme penalty
    flags = [{"score_impact": -120}]
    score_low = calculate_credibility_score(0.99, flags)
    assert score_low >= 0

    # Extreme bonus
    bonus_flags = [{"score_impact": +80}]
    score_high = calculate_credibility_score(0.01, bonus_flags)
    assert score_high <= 100


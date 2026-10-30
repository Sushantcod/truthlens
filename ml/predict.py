"""TruthLens Production Inference and Explainability Engine.

Implements:
1. TF-IDF feature extraction & ML Ensemble Inference
2. Knowledge-Based Reasoning & Credibility Rules Engine
3. Integrated Credibility Score (0 to 100)
4. Responsible AI Decision Logic (Uncertainty thresholding & Calibration)
5. Explainable AI (XAI) Word-Level Highlighting & Feature Attributions
"""

import logging
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import joblib
import numpy as np

from ml.features import extract_handcrafted_features
from ml.preprocess import clean_text, tokenize_and_lemmatize, preprocess_for_tfidf

logger = logging.getLogger("truthlens.predict")

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "ml" / "models" / "model.pkl"
VECTORIZER_PATH = BASE_DIR / "ml" / "models" / "vectorizer.pkl"

# Singleton cache for loaded artifacts
_CACHED_MODEL = None
_CACHED_VECTORIZER = None
_CACHED_FEATURE_NAMES = None
_CACHED_COEFS = None

# Known unreliable or fabricated news domains
UNRELIABLE_DOMAINS = {
    "infowars.com",
    "beforeitsnews.com",
    "worldnewsdailyreport.com",
    "naturalnews.com",
    "yournewswire.com",
    "newspunch.com",
    "thegatewaypundit.com",
    "neonnettle.com",
    "conservativetribune.com",
    "wakingtimes.com",
    "dailybuzzlive.com",
    "empirenews.net",
}

# Known reputable, peer-reviewed, or wire journalism domains
REPUTABLE_DOMAINS = {
    "reuters.com",
    "apnews.com",
    "bbc.com",
    "bbc.co.uk",
    "thehindu.com",
    "pti.in",
    "bloomberg.com",
    "nature.com",
    "science.org",
    "who.int",
    "nih.gov",
    "nasa.gov",
    "aljazeera.com",
    "theguardian.com",
    "npr.org",
    "pbs.org",
}

RESPONSIBLE_AI_DISCLAIMER = (
    "TruthLens AI provides algorithmic credibility assistance to aid human critical reading. "
    "Automated predictions should not replace primary-source verification or professional fact-checking."
)


def load_artifacts():
    """Load model and vectorizer singletons into memory."""
    global _CACHED_MODEL, _CACHED_VECTORIZER, _CACHED_FEATURE_NAMES, _CACHED_COEFS

    if _CACHED_MODEL is None or _CACHED_VECTORIZER is None:
        if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
            logger.info("Artifacts missing, invoking training pipeline...")
            from ml.train import train_models_pipeline
            train_models_pipeline()

        _CACHED_MODEL = joblib.load(MODEL_PATH)
        _CACHED_VECTORIZER = joblib.load(VECTORIZER_PATH)
        _CACHED_FEATURE_NAMES = _CACHED_VECTORIZER.get_feature_names_out()

        # Extract weights from first estimator (LR) inside voting classifier
        try:
            if hasattr(_CACHED_MODEL, "named_estimators_") and "lr" in _CACHED_MODEL.named_estimators_:
                _CACHED_COEFS = _CACHED_MODEL.named_estimators_["lr"].coef_[0]
            elif hasattr(_CACHED_MODEL, "coef_"):
                _CACHED_COEFS = _CACHED_MODEL.coef_[0]
            else:
                _CACHED_COEFS = np.zeros(len(_CACHED_FEATURE_NAMES))
        except Exception:
            _CACHED_COEFS = np.zeros(len(_CACHED_FEATURE_NAMES))

    return _CACHED_MODEL, _CACHED_VECTORIZER, _CACHED_FEATURE_NAMES, _CACHED_COEFS


def check_knowledge_rules(text: str, domain: Optional[str] = None) -> List[Dict[str, Any]]:
    """Execute rule-based heuristics against text and optional source domain."""
    flags: List[Dict[str, Any]] = []
    text_lower = text.lower()
    features = extract_handcrafted_features(text)

    # 1. Domain Credibility Check
    if domain:
        clean_domain = domain.lower().replace("www.", "").strip()
        if clean_domain in UNRELIABLE_DOMAINS:
            flags.append({
                "rule": "BLACKLISTED_UNRELIABLE_SOURCE",
                "severity": "HIGH",
                "score_impact": -35,
                "description": f"Domain '{clean_domain}' is documented on independent disinformation watchlists.",
            })
        elif clean_domain in REPUTABLE_DOMAINS:
            flags.append({
                "rule": "VERIFIED_REPUTABLE_SOURCE",
                "severity": "POSITIVE",
                "score_impact": +15,
                "description": f"Domain '{clean_domain}' is recognized as an institutional news wire or verified publisher.",
            })

    # 2. Excessive Capitalization
    if features["caps_ratio"] > 0.16 and features["word_count"] > 10:
        flags.append({
            "rule": "EXCESSIVE_CAPITALIZATION",
            "severity": "MEDIUM",
            "score_impact": -12,
            "description": f"Abnormally high uppercase ratio ({features['caps_ratio']*100:.1f}%), often used to simulate urgency.",
        })

    # 3. Exclamation Point Saturation
    if features["exclamation_ratio"] > 0.04:
        flags.append({
            "rule": "EXCLAMATION_SATURATION",
            "severity": "LOW",
            "score_impact": -8,
            "description": "Dense exclamation mark usage indicative of emotional manipulation rather than neutral reporting.",
        })

    # 4. Sensationalist / Clickbait Lexicon Hits
    if features["sensational_hits"] > 0:
        phrases = ", ".join(f"'{p}'" for p in features["matched_sensational_phrases"][:4])
        flags.append({
            "rule": "SENSATIONAL_LEXICAL_MARKERS",
            "severity": "HIGH" if features["sensational_hits"] >= 2 else "MEDIUM",
            "score_impact": -min(30, features["sensational_hits"] * 10),
            "description": f"Triggered sensational disinformation patterns: {phrases}.",
        })

    # 5. Missing Author / Source Attribution Heuristic
    attribution_markers = [
        "according to", "spokesperson", "statement released", "published in",
        "official said", "reported by", "press briefing", "documents show",
        "peer-reviewed", "confirmed by"
    ]
    has_attribution = any(m in text_lower for m in attribution_markers)
    if not has_attribution and features["word_count"] > 35:
        flags.append({
            "rule": "UNVERIFIED_ANONYMOUS_ASSERTION",
            "severity": "LOW",
            "score_impact": -10,
            "description": "Text lacks verifiable institutional attributions, quotes, or primary document citations.",
        })

    return flags


def calculate_credibility_score(fake_prob: float, flags: List[Dict[str, Any]]) -> int:
    """Combine ML probability and knowledge-base heuristic flags into a 0 to 100 score."""
    # Base factual credibility inversely mirrors the probability of deception
    base_credibility = (1.0 - fake_prob) * 100.0

    # Net rule adjustments
    rule_delta = sum(f.get("score_impact", 0) for f in flags)
    raw_score = base_credibility + rule_delta

    # Clip within valid range [0, 100]
    final_score = int(np.clip(round(raw_score), 0, 100))
    return final_score


def explain_prediction(text: str, top_k: int = 10) -> Tuple[List[Dict[str, Any]], str]:
    """Generate XAI word highlights and top weighted feature contributions."""
    _, vectorizer, feature_names, coefs = load_artifacts()

    # Preprocess text
    cleaned = preprocess_for_tfidf(text, remove_leakage=True)
    tokens = cleaned.split()

    feature_dict = {f: i for i, f in enumerate(feature_names)}

    # Identify tokens present and their linear weights
    token_weights: Dict[str, float] = {}
    for tok in tokens:
        if tok in feature_dict:
            idx = feature_dict[tok]
            token_weights[tok] = float(coefs[idx])

    # Top contributing features
    top_items = sorted(token_weights.items(), key=lambda x: abs(x[1]), reverse=True)[:top_k]
    top_features = [
        {
            "feature": k,
            "weight": round(v, 4),
            "signal": "FAKE" if v > 0 else "REAL",
            "direction": "Pushes toward FAKE" if v > 0 else "Pushes toward REAL",
        }
        for k, v in top_items
    ]

    # Build highlighted HTML output
    # Red (#e46c50) for FAKE indicators, Green (#287c70) for REAL indicators
    words = re.findall(r"\S+|\s+", text)
    highlighted_chunks = []

    for chunk in words:
        stripped = chunk.strip().lower().translate(str.maketrans("", "", r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""))
        if stripped in token_weights:
            w = token_weights[stripped]
            if w > 0.05:
                # Fake signal (coral/red)
                highlighted_chunks.append(f'<mark class="hl-fake" title="Fake signal (weight: +{w:.2f})">{chunk}</mark>')
            elif w < -0.05:
                # Real signal (teal/green)
                highlighted_chunks.append(f'<mark class="hl-real" title="Real signal (weight: {w:.2f})">{chunk}</mark>')
            else:
                highlighted_chunks.append(chunk)
        else:
            highlighted_chunks.append(chunk)

    highlighted_html = "".join(highlighted_chunks)
    return top_features, highlighted_html


def predict(text: str, domain: Optional[str] = None) -> Dict[str, Any]:
    """High-level prediction API combining ML inference, XAI, and rules.

    Args:
        text: Raw news text or headline.
        domain: Optional originating website domain.

    Returns:
        Structured prediction outcome with verdict, confidence, credibility score,
        rule flags, feature attributions, and responsible AI disclosures.
    """
    if not text or not text.strip():
        return {
            "verdict": "UNCERTAIN",
            "confidence": 0.0,
            "credibility_score": 50,
            "probabilities": {"REAL": 0.5, "FAKE": 0.5},
            "rule_flags": [],
            "top_features": [],
            "highlighted_html": "",
            "disclaimer": RESPONSIBLE_AI_DISCLAIMER,
        }

    model, vectorizer, _, _ = load_artifacts()

    # Preprocess text
    normalized_input = preprocess_for_tfidf(text, remove_leakage=True)
    X_vec = vectorizer.transform([normalized_input])

    # Infer probabilities: class 0 = REAL, class 1 = FAKE
    probs = model.predict_proba(X_vec)[0]
    p_real = float(probs[0])
    p_fake = float(probs[1])

    # Rule-based credibility checks
    flags = check_knowledge_rules(text, domain=domain)
    credibility_score = calculate_credibility_score(p_fake, flags)

    # Responsible AI Uncertainty Calibration:
    # If model confidence is below 60% (between 0.40 and 0.60 probability), flag UNCERTAIN
    if 0.40 <= p_fake <= 0.60:
        verdict = "UNCERTAIN"
        confidence = round(max(p_real, p_fake) * 100, 1)
        badge_color = "warning"
    elif p_fake > 0.60:
        verdict = "FAKE"
        confidence = round(p_fake * 100, 1)
        badge_color = "danger"
    else:
        verdict = "REAL"
        confidence = round(p_real * 100, 1)
        badge_color = "success"

    # Explainability: top features and highlighted text
    top_features, highlighted_html = explain_prediction(text, top_k=10)

    # Compute handcrafted features summary
    hc_features = extract_handcrafted_features(text)

    return {
        "verdict": verdict,
        "confidence": confidence,
        "credibility_score": credibility_score,
        "badge_color": badge_color,
        "probabilities": {
            "REAL": round(p_real, 4),
            "FAKE": round(p_fake, 4),
        },
        "rule_flags": flags,
        "top_features": top_features,
        "highlighted_html": highlighted_html,
        "handcrafted_signals": hc_features,
        "disclaimer": RESPONSIBLE_AI_DISCLAIMER,
    }
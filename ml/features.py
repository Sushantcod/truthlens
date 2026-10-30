"""TruthLens Feature Engineering Module.

Extracts stylometric, emotional, and structural signals from text:
1. Text Length and Word Count
2. Uppercase (Caps) Ratio
3. Exclamation and Question Mark Density
4. Clickbait and Sensational Lexical Frequency
5. Sentiment Polarity & Subjectivity (Lexicon / VADER-compatible)
6. Readability Metric (Flesch-Kincaid Grade Level approximation)
"""

import math
import re
import string
from typing import Dict, Any, List
import numpy as np

# Sensational, conspiratorial, and clickbait lexical patterns
SENSATIONAL_PATTERNS = [
    r"\bshocking\b",
    r"\bbombshell\b",
    r"\bunbelievable\b",
    r"\byou won't believe\b",
    r"\bsecret\b",
    r"\bexposed\b",
    r"\bwake up\b",
    r"\bsheeple\b",
    r"\bshare before it's deleted\b",
    r"\bmiracle cure\b",
    r"\bhidden truth\b",
    r"\bwhat they don't want you to know\b",
    r"\bmainstream media is lying\b",
    r"\b100% cure\b",
    r"\bconspiracy\b",
    r"\bdeep state\b",
    r"\bcorrupt cartel\b",
    r"\burgent alert\b",
    r"\bforbidden\b",
    r"\bclassified leak\b",
    r"\bproven hoax\b",
]

COMPILED_SENSATIONAL = [re.compile(p, re.IGNORECASE) for p in SENSATIONAL_PATTERNS]

# Lexicons for sentiment polarity estimation
POSITIVE_WORDS = {
    "agreement", "alliance", "approved", "benefit", "breakthrough", "collaborative",
    "consensus", "cure", "effective", "expansion", "growth", "improvement",
    "innovation", "peace", "progress", "recovery", "resilient", "success",
    "sustainable", "treaty", "unanimous", "verified", "victory", "vital"
}

NEGATIVE_WORDS = {
    "attack", "blackout", "catastrophe", "collapse", "conspiracy", "corrupt",
    "crisis", "danger", "deadly", "deceit", "destroy", "disaster",
    "epidemic", "evil", "fake", "fatal", "fraud", "hoax", "illegal",
    "panic", "poison", "refuses", "scandal", "terrifying", "toxic", "threat"
}


def count_syllables(word: str) -> int:
    """Approximate syllable count in an English word."""
    word = word.lower().strip(".:;?!")
    if not word:
        return 0
    if len(word) <= 3:
        return 1
    # Count vowel groups
    count = len(re.findall(r"[aeiouy]+", word))
    # Adjust for silent 'e' at end
    if word.endswith("e") and not word.endswith("le"):
        count = max(1, count - 1)
    return max(1, count)


def compute_readability_flesch_kincaid(text: str) -> float:
    """Compute Flesch-Kincaid Grade Level metric.

    Formula: 0.39 * (total_words / total_sentences) + 11.8 * (total_syllables / total_words) - 15.59
    Typical values range from 1 to 18+. Higher grade levels indicate denser, academic/official text.
    """
    if not text or not text.strip():
        return 0.0

    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    words = re.findall(r"\b[A-Za-z]+\b", text)

    num_sentences = max(1, len(sentences))
    num_words = max(1, len(words))

    num_syllables = sum(count_syllables(w) for w in words)

    asl = num_words / num_sentences  # Average Sentence Length
    asw = num_syllables / num_words  # Average Syllables per Word

    score = 0.39 * asl + 11.8 * asw - 15.59
    return round(float(np.clip(score, 0.0, 24.0)), 2)


def compute_sentiment_polarity(text: str) -> float:
    """Compute a normalized sentiment polarity score from -1.0 (very negative) to +1.0 (very positive)."""
    words = [w.lower().strip(string.punctuation) for w in text.split()]
    if not words:
        return 0.0

    pos_count = sum(1 for w in words if w in POSITIVE_WORDS)
    neg_count = sum(1 for w in words if w in NEGATIVE_WORDS)
    total = pos_count + neg_count

    if total == 0:
        return 0.0

    polarity = (pos_count - neg_count) / total
    return round(float(polarity), 3)


def extract_handcrafted_features(text: str) -> Dict[str, Any]:
    """Extract full dictionary of stylometric and linguistic features from raw text.

    Args:
        text: Raw input news text.

    Returns:
        Dictionary of computed feature values.
    """
    if not text:
        text = ""

    total_chars = len(text)
    words = text.split()
    total_words = len(words)

    # 1. Uppercase letter ratio
    alpha_chars = sum(1 for c in text if c.isalpha())
    upper_chars = sum(1 for c in text if c.isupper())
    caps_ratio = (upper_chars / alpha_chars) if alpha_chars > 0 else 0.0

    # 2. Exclamation mark count & ratio
    excl_count = text.count("!")
    excl_ratio = (excl_count / total_words) if total_words > 0 else 0.0

    # 3. Question mark count & ratio
    ques_count = text.count("?")
    ques_ratio = (ques_count / total_words) if total_words > 0 else 0.0

    # 4. Sensational / Clickbait lexical occurrences
    sensational_hits = 0
    matched_sensational_phrases: List[str] = []
    text_lower = text.lower()
    for pattern in COMPILED_SENSATIONAL:
        matches = pattern.findall(text_lower)
        if matches:
            sensational_hits += len(matches)
            matched_sensational_phrases.append(pattern.pattern.replace(r"\b", ""))

    sensational_density = (sensational_hits / max(1, total_words)) * 100.0

    # 5. Readability & Sentiment
    readability_grade = compute_readability_flesch_kincaid(text)
    sentiment_polarity = compute_sentiment_polarity(text)

    # 6. Average word length
    avg_word_len = sum(len(w) for w in words) / total_words if total_words > 0 else 0.0

    return {
        "char_count": total_chars,
        "word_count": total_words,
        "caps_ratio": round(caps_ratio, 4),
        "exclamation_count": excl_count,
        "exclamation_ratio": round(excl_ratio, 4),
        "question_count": ques_count,
        "question_ratio": round(ques_ratio, 4),
        "sensational_hits": sensational_hits,
        "sensational_density": round(sensational_density, 3),
        "matched_sensational_phrases": matched_sensational_phrases,
        "readability_grade": readability_grade,
        "sentiment_polarity": sentiment_polarity,
        "avg_word_length": round(avg_word_len, 2),
    }


def get_feature_vector(text: str) -> np.ndarray:
    """Return numeric array of handcrafted features suitable for scikit-learn models."""
    feat = extract_handcrafted_features(text)
    return np.array([
        feat["caps_ratio"],
        feat["exclamation_ratio"],
        feat["question_ratio"],
        feat["sensational_density"],
        feat["readability_grade"] / 20.0,  # Scaled roughly 0..1
        feat["sentiment_polarity"],
        min(1.0, feat["word_count"] / 1000.0),
    ], dtype=np.float32)


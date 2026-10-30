"""TruthLens Natural Language Preprocessing and Anti-Leakage Pipeline.

=============================================================================
CRITICAL NOTE ON PREVENTING DATA LEAKAGE:
Many public fake news datasets (notably Kaggle ISOT) contain systematic artifact
leakage: almost every real news article begins with 'WASHINGTON (Reuters) -',
'(Reuters)', or mentions 'Reuters' in the first paragraph, whereas fake news
articles frequently contain publisher tags or Twitter handles.
A naive model trained without removing these signatures learns trivial heuristics
(e.g., presence of the word 'Reuters' => 99.8% REAL) rather than semantic and
stylistic credibility markers. This leads to severe catastrophic failure when
evaluating real-world or out-of-domain articles.

This module explicitly strips all wire signatures, datelines, and source artifacts
before text representation, ensuring generalizable representation learning.
=============================================================================
"""

import re
import string
from typing import List, Optional
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure NLTK data dependencies
try:
    STOP_WORDS = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords", quiet=True)
    STOP_WORDS = set(stopwords.words("english"))

try:
    LEMMATIZER = WordNetLemmatizer()
    LEMMATIZER.lemmatize("testing")
except LookupError:
    nltk.download("wordnet", quiet=True)
    LEMMATIZER = WordNetLemmatizer()

# Wire service and publisher leakage patterns
LEAKAGE_PATTERNS = [
    # Dateline stamps like "WASHINGTON (Reuters) - " or "GENEVA (Reuters) -"
    r"^[A-Z\s]{2,20}\s*\((?:Reuters|AP|AFP|Bloomberg|PTI)\)\s*[-–—]\s*",
    r"\((?:Reuters|AP|AFP|Bloomberg|PTI)\)",
    r"(?:Reuters|Associated Press|Agence France-Presse)\s*[-–—]\s*",
    r"^[A-Z\s]{2,20}\s*[-–—]\s*",
    # Social media handles and hashtags
    r"@[\w_]+",
    r"#[\w_]+",
    # Specific ISOT and disinformation collection signatures
    r"Featured image via\s+.*$",
    r"Image credit:?\s+.*$",
]

COMPILED_LEAKAGE_REGEX = [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in LEAKAGE_PATTERNS]
URL_REGEX = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
HTML_REGEX = re.compile(r"<.*?>")
MULTIPLE_SPACES_REGEX = re.compile(r"\s+")
PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def strip_data_leakage(text: str) -> str:
    """Remove wire service datelines, publisher watermarks, and source signatures.

    Args:
        text: Raw news text.

    Returns:
        Cleaned text stripped of publisher-identifying tokens.
    """
    if not text or not isinstance(text, str):
        return ""

    cleaned = text
    for pattern in COMPILED_LEAKAGE_REGEX:
        cleaned = pattern.sub(" ", cleaned)

    return cleaned


def clean_text(text: str, remove_leakage: bool = True) -> str:
    """Full normalization: strips URLs, HTML, publisher leakage, and excess whitespace.

    Args:
        text: Input string.
        remove_leakage: Whether to strip publisher dateline markers.

    Returns:
        Normalized clean string.
    """
    if not text or not isinstance(text, str):
        return ""

    # Remove URLs and HTML tags
    cleaned = URL_REGEX.sub(" ", text)
    cleaned = HTML_REGEX.sub(" ", cleaned)

    if remove_leakage:
        cleaned = strip_data_leakage(cleaned)

    # Collapse multiple whitespaces
    cleaned = MULTIPLE_SPACES_REGEX.sub(" ", cleaned).strip()
    return cleaned


def tokenize_and_lemmatize(text: str, remove_stopwords: bool = True) -> List[str]:
    """Tokenize, remove punctuation and numbers, filter stopwords, and lemmatize tokens.

    Args:
        text: Cleaned input text.
        remove_stopwords: Whether to remove common English stopwords.

    Returns:
        List of lemmatized, normalized token strings.
    """
    if not text or not isinstance(text, str):
        return []

    # Lowercase and strip punctuation
    text_clean = text.lower().translate(PUNCT_TABLE)
    raw_tokens = text_clean.split()

    tokens: List[str] = []
    for tok in raw_tokens:
        tok_clean = tok.strip()
        if not tok_clean or tok_clean.isnumeric() or len(tok_clean) < 2:
            continue
        if remove_stopwords and tok_clean in STOP_WORDS:
            continue
        # Apply WordNet lemmatization
        lemma = LEMMATIZER.lemmatize(tok_clean)
        tokens.append(lemma)

    return tokens


def preprocess_for_tfidf(text: str, remove_leakage: bool = True) -> str:
    """Convenience string pipeline for scikit-learn TF-IDF vectorizers.

    Args:
        text: Raw document text.
        remove_leakage: Enable anti-leakage filters.

    Returns:
        Space-separated string of lemmatized, stopword-filtered tokens.
    """
    cleaned = clean_text(text, remove_leakage=remove_leakage)
    tokens = tokenize_and_lemmatize(cleaned, remove_stopwords=True)
    return " ".join(tokens)


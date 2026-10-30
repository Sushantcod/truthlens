"""TruthLens Knowledge Rules Service Facade."""

from typing import List, Dict, Any, Optional
from ml.predict import check_knowledge_rules, UNRELIABLE_DOMAINS, REPUTABLE_DOMAINS


def evaluate_rules(text: str, domain: Optional[str] = None) -> List[Dict[str, Any]]:
    """Evaluate text and domain against expert knowledge rules."""
    return check_knowledge_rules(text, domain=domain)


def is_known_unreliable(domain: str) -> bool:
    """Check if domain is on known disinformation watchlist."""
    if not domain:
        return False
    return domain.lower().replace("www.", "").strip() in UNRELIABLE_DOMAINS


def is_known_reputable(domain: str) -> bool:
    """Check if domain is an institutional wire / publisher."""
    if not domain:
        return False
    return domain.lower().replace("www.", "").strip() in REPUTABLE_DOMAINS


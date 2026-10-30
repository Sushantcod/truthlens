"""TruthLens Article URL Scraper Service.

Robustly extracts headlines, article body content, and canonical domain
metadata from live web pages using BeautifulSoup4 with defensive fallbacks,
timeouts, and header spoofing to prevent automated bot blocks.
"""

import logging
import urllib.parse
from typing import Dict, Any, Optional
from bs4 import BeautifulSoup
import requests

logger = logging.getLogger("truthlens.scraper")

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36 TruthLens/1.0"
)


def extract_domain(url: str) -> str:
    """Extract clean domain name from URL."""
    try:
        parsed = urllib.parse.urlparse(url)
        domain = parsed.netloc.lower()
        if domain.startswith("www."):
            domain = domain[4:]
        return domain
    except Exception:
        return ""


def scrape_article_from_url(url: str, timeout: int = 8) -> Dict[str, Any]:
    """Scrape headline and text from given URL.

    Args:
        url: Full web URL.
        timeout: Request timeout in seconds.

    Returns:
        Dict with status, title, text, domain, or error message.
    """
    domain = extract_domain(url)
    if not domain:
        return {"success": False, "error": "Invalid URL provided."}

    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }

    try:
        response = requests.get(url, headers=headers, timeout=timeout)
        if response.status_code != 200:
            return {
                "success": False,
                "error": f"Target server returned HTTP status {response.status_code}.",
                "domain": domain,
            }

        soup = BeautifulSoup(response.text, "html.parser")

        # 1. Extract Title
        title = ""
        og_title = soup.find("meta", property="og:title")
        if og_title and og_title.get("content"):
            title = og_title["content"].strip()
        elif soup.find("h1"):
            title = soup.find("h1").get_text(strip=True)
        elif soup.title:
            title = soup.title.get_text(strip=True)

        # 2. Extract Body Text
        # Strip script, style, navigation, footer tags
        for element in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
            element.decompose()

        # Target article container or collect all substantial paragraphs
        article_elem = soup.find("article")
        if article_elem:
            paragraphs = article_elem.find_all("p")
        else:
            paragraphs = soup.find_all("p")

        text_blocks = [p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 25]
        extracted_text = "\n\n".join(text_blocks)

        if not extracted_text:
            return {
                "success": False,
                "error": "Could not extract legible article paragraphs from this webpage.",
                "domain": domain,
                "title": title,
            }

        return {
            "success": True,
            "title": title or "Scraped Article",
            "text": extracted_text,
            "domain": domain,
            "url": url,
        }

    except requests.exceptions.Timeout:
        return {"success": False, "error": "Connection timed out while fetching article.", "domain": domain}
    except requests.exceptions.RequestException as e:
        return {"success": False, "error": f"Failed to retrieve URL: {str(e)}", "domain": domain}
    except Exception as e:
        logger.exception("Unexpected error during scraping")
        return {"success": False, "error": f"Parsing failed: {str(e)}", "domain": domain}


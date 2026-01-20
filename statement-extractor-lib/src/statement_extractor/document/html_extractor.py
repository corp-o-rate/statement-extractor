"""
HTML text extraction utilities.

Extracts clean text content from HTML pages, prioritizing article content
and removing navigation, headers, footers, and other non-content elements.
"""

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)


def extract_text_from_html(html: str) -> tuple[str, Optional[str]]:
    """
    Extract clean text and title from HTML.

    Removes scripts, styles, navigation, and other non-content elements.
    Prioritizes article or main content areas.

    Args:
        html: Raw HTML string

    Returns:
        Tuple of (extracted_text, title or None)
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        raise ImportError(
            "BeautifulSoup is required for HTML extraction. "
            "Install with: pip install beautifulsoup4"
        )

    soup = BeautifulSoup(html, "html.parser")

    # Remove unwanted elements
    for tag_name in [
        "script",
        "style",
        "nav",
        "footer",
        "header",
        "aside",
        "noscript",
        "iframe",
        "form",
        "button",
        "input",
        "select",
        "textarea",
    ]:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    # Remove elements with common non-content class/id patterns
    non_content_patterns = [
        "nav",
        "menu",
        "sidebar",
        "footer",
        "header",
        "comment",
        "advertisement",
        "ad-",
        "social",
        "share",
        "related",
        "recommended",
        "popup",
        "modal",
        "cookie",
        "banner",
        "promo",
    ]

    # Collect elements to remove first, then decompose
    # (decomposing while iterating can cause issues)
    elements_to_remove = []

    for element in soup.find_all(class_=True):
        if element.attrs is None:
            continue
        classes = element.get("class", [])
        if classes:
            class_str = " ".join(classes).lower()
            if any(pattern in class_str for pattern in non_content_patterns):
                elements_to_remove.append(element)

    for element in soup.find_all(id=True):
        if element.attrs is None:
            continue
        element_id = element.get("id", "")
        if element_id and any(pattern in element_id.lower() for pattern in non_content_patterns):
            elements_to_remove.append(element)

    for element in elements_to_remove:
        try:
            element.decompose()
        except Exception:
            pass  # Element may already be decomposed

    # Get title
    title = None
    if soup.title and soup.title.string:
        title = soup.title.string.strip()
        # Clean up common title patterns (e.g., "Article Title | Site Name")
        title = re.split(r"\s*[|—\-]\s*", title)[0].strip()

    # Find main content area
    content = None

    # Priority: article > main > [role="main"] > body
    for selector in ["article", "main", "[role='main']", ".content", "#content"]:
        content = soup.select_one(selector)
        if content and len(content.get_text(strip=True)) > 100:
            break

    if not content:
        content = soup.body or soup

    # Extract text using BeautifulSoup's get_text with newline separator
    text = content.get_text(separator="\n", strip=True)

    # Clean up whitespace
    text = _clean_whitespace(text)

    logger.debug(f"Extracted {len(text)} chars from HTML (title: {title})")

    return text, title


def _clean_whitespace(text: str) -> str:
    """
    Clean up whitespace while preserving paragraph structure.

    Args:
        text: Raw extracted text

    Returns:
        Cleaned text
    """
    # Normalize line breaks
    text = re.sub(r"\r\n?", "\n", text)

    # Collapse multiple spaces (but not newlines)
    text = re.sub(r"[^\S\n]+", " ", text)

    # Collapse multiple newlines to max 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove leading/trailing whitespace from each line
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)

    # Remove empty lines at start/end
    text = text.strip()

    return text


def extract_article_content(html: str) -> tuple[str, dict]:
    """
    Extract article content with metadata.

    Attempts to extract structured article data including:
    - Title
    - Author
    - Published date
    - Main content

    Args:
        html: Raw HTML string

    Returns:
        Tuple of (content, metadata dict)
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        raise ImportError(
            "BeautifulSoup is required for HTML extraction. "
            "Install with: pip install beautifulsoup4"
        )

    soup = BeautifulSoup(html, "html.parser")

    metadata = {}

    # Extract title
    title = None
    # Try og:title first
    og_title = soup.find("meta", property="og:title")
    if og_title and og_title.get("content"):
        title = og_title["content"].strip()
    elif soup.title and soup.title.string:
        title = soup.title.string.strip()

    if title:
        metadata["title"] = title

    # Extract author
    author = None
    author_meta = soup.find("meta", attrs={"name": "author"})
    if author_meta and author_meta.get("content"):
        author = author_meta["content"].strip()
    else:
        # Try common author class patterns
        author_elem = soup.select_one(".author, .byline, [rel='author']")
        if author_elem:
            author = author_elem.get_text(strip=True)

    if author:
        metadata["author"] = author

    # Extract published date
    date = None
    date_meta = soup.find("meta", property="article:published_time")
    if date_meta and date_meta.get("content"):
        date = date_meta["content"]
    else:
        date_elem = soup.select_one("time[datetime], .date, .published")
        if date_elem:
            date = date_elem.get("datetime") or date_elem.get_text(strip=True)

    if date:
        metadata["published_date"] = date

    # Extract description
    description = None
    desc_meta = soup.find("meta", attrs={"name": "description"})
    if desc_meta and desc_meta.get("content"):
        description = desc_meta["content"].strip()

    if description:
        metadata["description"] = description

    # Extract main content
    content, extracted_title = extract_text_from_html(html)

    # Use extracted title if we didn't find one
    if not title and extracted_title:
        metadata["title"] = extracted_title

    return content, metadata

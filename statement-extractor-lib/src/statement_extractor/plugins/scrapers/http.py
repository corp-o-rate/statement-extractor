"""
HTTP scraper plugin for fetching web content.

Uses httpx for async HTTP requests with retries, timeouts, and CAPTCHA detection.
"""

import logging
from typing import Optional

from ..base import BaseScraperPlugin, ContentType, ScraperResult
from ...pipeline.registry import PluginRegistry

logger = logging.getLogger(__name__)


@PluginRegistry.scraper
class HttpScraperPlugin(BaseScraperPlugin):
    """
    Default HTTP scraper using httpx with retries and timeouts.

    Features:
    - Async HTTP requests with httpx
    - Automatic redirect following
    - Content type detection from headers and URL
    - CAPTCHA page detection
    - Configurable timeout and retries
    """

    def __init__(
        self,
        timeout: float = 30.0,
        max_retries: int = 3,
        user_agent: str = "Mozilla/5.0 (compatible; StatementExtractor/1.0; +https://github.com/corp-o-rate/statement-extractor)",
        follow_redirects: bool = True,
    ):
        self._timeout = timeout
        self._max_retries = max_retries
        self._user_agent = user_agent
        self._follow_redirects = follow_redirects

    @property
    def name(self) -> str:
        return "http_scraper"

    @property
    def priority(self) -> int:
        return 100  # Default scraper

    @property
    def description(self) -> str:
        return "Default HTTP scraper using httpx with retries and CAPTCHA detection"

    async def fetch(self, url: str, timeout: Optional[float] = None) -> ScraperResult:
        """
        Fetch content from a URL with retries and CAPTCHA detection.

        Args:
            url: The URL to fetch
            timeout: Request timeout in seconds (uses instance default if None)

        Returns:
            ScraperResult with content, content type, and any errors
        """
        import httpx

        timeout = timeout or self._timeout
        last_error: Optional[str] = None

        for attempt in range(self._max_retries):
            try:
                async with httpx.AsyncClient(
                    timeout=timeout,
                    follow_redirects=self._follow_redirects,
                ) as client:
                    logger.debug(f"Fetching URL: {url} (attempt {attempt + 1})")

                    response = await client.get(
                        url,
                        headers={"User-Agent": self._user_agent},
                    )

                    content_type = self._detect_content_type(
                        dict(response.headers), url
                    )

                    # Check for CAPTCHA if HTML
                    error = None
                    if content_type == ContentType.HTML:
                        if self._is_captcha_page(response.content):
                            error = "CAPTCHA or challenge page detected"
                            logger.warning(f"CAPTCHA detected at {url}")

                    return ScraperResult(
                        url=url,
                        final_url=str(response.url),
                        content=response.content,
                        content_type=content_type,
                        headers=dict(response.headers),
                        error=error,
                    )

            except httpx.TimeoutException as e:
                last_error = f"Request timed out after {timeout}s"
                logger.warning(f"Timeout fetching {url}: {e}")
            except httpx.ConnectError as e:
                last_error = f"Connection error: {e}"
                logger.warning(f"Connection error fetching {url}: {e}")
            except httpx.HTTPStatusError as e:
                last_error = f"HTTP {e.response.status_code}: {e.response.reason_phrase}"
                logger.warning(f"HTTP error fetching {url}: {e}")
                # Don't retry on 4xx errors
                if 400 <= e.response.status_code < 500:
                    break
            except Exception as e:
                last_error = f"Unexpected error: {e}"
                logger.exception(f"Error fetching {url}")

        # All retries failed
        return ScraperResult(
            url=url,
            final_url=url,
            content=b"",
            content_type=ContentType.UNKNOWN,
            error=last_error or "Unknown error",
        )

    async def head(self, url: str, timeout: Optional[float] = None) -> ScraperResult:
        """
        Check content type without downloading the full body.

        Args:
            url: The URL to check
            timeout: Request timeout in seconds

        Returns:
            ScraperResult with content_type populated (content is empty)
        """
        import httpx

        timeout = timeout or self._timeout

        try:
            async with httpx.AsyncClient(
                timeout=timeout,
                follow_redirects=self._follow_redirects,
            ) as client:
                response = await client.head(
                    url,
                    headers={"User-Agent": self._user_agent},
                )

                content_type = self._detect_content_type(
                    dict(response.headers), url
                )

                return ScraperResult(
                    url=url,
                    final_url=str(response.url),
                    content=b"",
                    content_type=content_type,
                    headers=dict(response.headers),
                )

        except Exception as e:
            logger.warning(f"HEAD request failed for {url}: {e}")
            # Fall back to full fetch
            return await self.fetch(url, timeout)

    @staticmethod
    def _detect_content_type(headers: dict[str, str], url: str) -> ContentType:
        """
        Detect content type from HTTP headers and URL.

        Priority:
        1. Content-Type header
        2. URL file extension
        """
        content_type_header = headers.get("content-type", "").lower()

        # Check Content-Type header
        if "application/pdf" in content_type_header:
            return ContentType.PDF
        if any(mime in content_type_header for mime in [
            "text/html",
            "application/xhtml+xml",
        ]):
            return ContentType.HTML

        # Check URL extension
        url_lower = url.lower().split("?")[0]  # Remove query params
        if url_lower.endswith(".pdf"):
            return ContentType.PDF
        if url_lower.endswith((".html", ".htm")):
            return ContentType.HTML

        # Default based on content-type
        if content_type_header.startswith("text/"):
            return ContentType.HTML
        if content_type_header.startswith(("image/", "audio/", "video/")):
            return ContentType.BINARY

        return ContentType.UNKNOWN

    @staticmethod
    def _is_captcha_page(content: bytes) -> bool:
        """
        Detect CAPTCHA or challenge pages.

        Checks for common CAPTCHA patterns in HTML content.
        """
        try:
            html = content.decode("utf-8", errors="replace").lower()
        except Exception:
            return False

        # Only check small pages (challenge pages are usually small)
        if len(html) > 50000:
            return False

        # Common CAPTCHA/challenge indicators
        captcha_patterns = [
            "captcha",
            "cloudflare",
            "checking your browser",
            "please verify you are a human",
            "access denied",
            "bot protection",
            "ddos protection",
            "just a moment",
            "enable javascript",
            "please enable cookies",
            "verify you are human",
            "security check",
        ]

        return any(pattern in html for pattern in captcha_patterns)

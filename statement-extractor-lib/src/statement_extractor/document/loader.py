"""
URL loader for fetching and parsing web content.

Orchestrates scraper and PDF parser plugins to load documents from URLs.
"""

import asyncio
import logging
from typing import Optional

from pydantic import BaseModel, Field

from ..models.document import Document
from ..pipeline.registry import PluginRegistry
from ..plugins.base import (
    BaseScraperPlugin,
    BasePDFParserPlugin,
    ContentType,
    ScraperResult,
)
from .html_extractor import extract_text_from_html, extract_article_content

logger = logging.getLogger(__name__)


class URLLoaderConfig(BaseModel):
    """Configuration for URL loading."""

    timeout: float = Field(
        default=30.0,
        description="Request timeout in seconds"
    )
    use_ocr: bool = Field(
        default=False,
        description="Force OCR for PDF parsing"
    )
    max_pdf_pages: int = Field(
        default=500,
        description="Maximum pages to extract from PDFs"
    )
    scraper_plugin: Optional[str] = Field(
        default=None,
        description="Specific scraper plugin to use (None = auto-select)"
    )
    pdf_parser_plugin: Optional[str] = Field(
        default=None,
        description="Specific PDF parser plugin to use (None = auto-select)"
    )
    extract_metadata: bool = Field(
        default=True,
        description="Extract article metadata from HTML pages"
    )


class URLLoader:
    """
    Loads documents from URLs using scraper and PDF parser plugins.

    Orchestrates the content acquisition process:
    1. Fetch content using a scraper plugin
    2. Detect content type (HTML vs PDF)
    3. Parse content using appropriate parser
    4. Create a Document object

    Example:
        >>> loader = URLLoader()
        >>> document = await loader.load("https://example.com/article")
        >>> print(document.title)

        >>> # Synchronous usage
        >>> document = loader.load_sync("https://example.com/report.pdf")
    """

    def __init__(self, config: Optional[URLLoaderConfig] = None):
        """
        Initialize the URL loader.

        Args:
            config: Loader configuration
        """
        self.config = config or URLLoaderConfig()
        self._scraper: Optional[BaseScraperPlugin] = None
        self._pdf_parser: Optional[BasePDFParserPlugin] = None

    def _get_scraper(self) -> BaseScraperPlugin:
        """Get the scraper plugin to use."""
        if self._scraper is not None:
            return self._scraper

        scrapers = PluginRegistry.get_scrapers()
        if not scrapers:
            raise RuntimeError(
                "No scraper plugins registered. "
                "Ensure plugins are loaded via 'from statement_extractor import plugins'"
            )

        if self.config.scraper_plugin:
            for scraper in scrapers:
                if scraper.name == self.config.scraper_plugin:
                    self._scraper = scraper
                    return scraper
            raise ValueError(f"Scraper plugin not found: {self.config.scraper_plugin}")

        # Use first available (highest priority)
        self._scraper = scrapers[0]
        return self._scraper

    def _get_pdf_parser(self) -> BasePDFParserPlugin:
        """Get the PDF parser plugin to use."""
        if self._pdf_parser is not None:
            return self._pdf_parser

        parsers = PluginRegistry.get_pdf_parsers()
        if not parsers:
            raise RuntimeError(
                "No PDF parser plugins registered. "
                "Ensure plugins are loaded via 'from statement_extractor import plugins'"
            )

        if self.config.pdf_parser_plugin:
            for parser in parsers:
                if parser.name == self.config.pdf_parser_plugin:
                    self._pdf_parser = parser
                    return parser
            raise ValueError(f"PDF parser plugin not found: {self.config.pdf_parser_plugin}")

        # Use first available (highest priority)
        self._pdf_parser = parsers[0]
        return self._pdf_parser

    async def load(self, url: str) -> Document:
        """
        Load a URL and return a Document.

        Args:
            url: URL to load

        Returns:
            Document with extracted content

        Raises:
            ValueError: If URL cannot be fetched or parsed
        """
        logger.info(f"Loading URL: {url}")

        # 1. Fetch content
        scraper = self._get_scraper()
        result = await scraper.fetch(url, self.config.timeout)

        if not result.ok:
            raise ValueError(f"Failed to fetch {url}: {result.error}")

        logger.debug(f"Fetched {len(result.content)} bytes, type: {result.content_type}")

        # 2. Process based on content type
        if result.content_type == ContentType.PDF:
            return self._process_pdf(result)
        elif result.content_type == ContentType.HTML:
            return self._process_html(result)
        else:
            # Try to guess based on content
            if result.content[:5] == b"%PDF-":
                return self._process_pdf(result)
            # Default to HTML
            return self._process_html(result)

    def load_sync(self, url: str) -> Document:
        """
        Synchronous wrapper for load().

        Args:
            url: URL to load

        Returns:
            Document with extracted content
        """
        return asyncio.run(self.load(url))

    def _process_pdf(self, result: ScraperResult) -> Document:
        """
        Convert PDF to Document with pages.

        Args:
            result: ScraperResult containing PDF bytes

        Returns:
            Document with PDF content
        """
        logger.info(f"Processing PDF from {result.final_url}")

        parser = self._get_pdf_parser()
        parse_result = parser.parse(
            result.content,
            max_pages=self.config.max_pdf_pages,
            use_ocr=self.config.use_ocr,
        )

        if not parse_result.ok:
            raise ValueError(f"Failed to parse PDF: {parse_result.error}")

        logger.info(f"Extracted {len(parse_result.pages)} pages from PDF")

        # Create Document from pages
        kwargs = {
            "pages": parse_result.pages,
            "title": parse_result.metadata.get("title"),
            "source_type": "pdf",
            "url": result.final_url,
        }
        author = parse_result.metadata.get("author")
        if author:
            kwargs["authors"] = [author]

        return Document.from_pages(**kwargs)

    def _process_html(self, result: ScraperResult) -> Document:
        """
        Convert HTML to Document (single page).

        Args:
            result: ScraperResult containing HTML bytes

        Returns:
            Document with HTML content
        """
        logger.info(f"Processing HTML from {result.final_url}")

        # Decode HTML
        try:
            html = result.content.decode("utf-8", errors="replace")
        except Exception as e:
            raise ValueError(f"Failed to decode HTML: {e}")

        # Extract text and metadata
        if self.config.extract_metadata:
            text, metadata = extract_article_content(html)
            title = metadata.get("title")
            author = metadata.get("author")
            # Log extracted metadata
            logger.debug(f"Extracted metadata: {metadata}")
        else:
            text, title = extract_text_from_html(html)
            author = None
            metadata = {}

        if not text or len(text.strip()) < 50:
            raise ValueError("No meaningful content extracted from HTML")

        logger.info(f"Extracted {len(text)} chars from HTML")
        if title:
            logger.info(f"  Title: {title}")
        if author:
            logger.info(f"  Author: {author}")
        if metadata.get("published_date"):
            logger.info(f"  Published: {metadata.get('published_date')}")

        # Create Document using from_pages since from_text forces source_type="text"
        kwargs = {
            "pages": [text],
            "title": title,
            "source_type": "webpage",
            "url": result.final_url,
        }
        if author:
            kwargs["authors"] = [author]

        return Document.from_pages(**kwargs)


async def load_url(
    url: str,
    config: Optional[URLLoaderConfig] = None,
) -> Document:
    """
    Convenience function to load a URL.

    Args:
        url: URL to load
        config: Optional loader configuration

    Returns:
        Document with extracted content
    """
    loader = URLLoader(config)
    return await loader.load(url)


def load_url_sync(
    url: str,
    config: Optional[URLLoaderConfig] = None,
) -> Document:
    """
    Convenience function to load a URL synchronously.

    Args:
        url: URL to load
        config: Optional loader configuration

    Returns:
        Document with extracted content
    """
    loader = URLLoader(config)
    return loader.load_sync(url)

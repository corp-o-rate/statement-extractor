"""
Tests for URL processing functionality.

Tests the scraper plugins, PDF parser plugins, HTML extractor, and URL loader.
"""

import pytest

from statement_extractor.plugins.base import (
    ContentType,
    ScraperResult,
    PDFParseResult,
    BaseScraperPlugin,
    BasePDFParserPlugin,
)
from statement_extractor.document.html_extractor import (
    extract_text_from_html,
    extract_article_content,
)


class TestContentType:
    """Tests for ContentType enum."""

    def test_content_types(self):
        """Test all content types exist."""
        assert ContentType.HTML == "html"
        assert ContentType.PDF == "pdf"
        assert ContentType.BINARY == "binary"
        assert ContentType.UNKNOWN == "unknown"


class TestScraperResult:
    """Tests for ScraperResult model."""

    def test_ok_success(self):
        """Test ok property for successful result."""
        result = ScraperResult(
            url="https://example.com",
            final_url="https://example.com",
            content=b"<html>content</html>",
            content_type=ContentType.HTML,
        )
        assert result.ok is True

    def test_ok_error(self):
        """Test ok property when error is set."""
        result = ScraperResult(
            url="https://example.com",
            final_url="https://example.com",
            content=b"",
            content_type=ContentType.UNKNOWN,
            error="Connection failed",
        )
        assert result.ok is False

    def test_ok_empty_content(self):
        """Test ok property with empty content."""
        result = ScraperResult(
            url="https://example.com",
            final_url="https://example.com",
            content=b"",
            content_type=ContentType.HTML,
        )
        assert result.ok is False


class TestPDFParseResult:
    """Tests for PDFParseResult model."""

    def test_ok_success(self):
        """Test ok property for successful result."""
        result = PDFParseResult(
            pages=["Page 1 content", "Page 2 content"],
            page_count=2,
        )
        assert result.ok is True

    def test_ok_error(self):
        """Test ok property when error is set."""
        result = PDFParseResult(
            pages=[],
            page_count=0,
            error="Failed to parse PDF",
        )
        assert result.ok is False

    def test_full_text(self):
        """Test full_text property."""
        result = PDFParseResult(
            pages=["Page 1", "Page 2", "Page 3"],
            page_count=3,
        )
        assert result.full_text == "Page 1\n\nPage 2\n\nPage 3"


class TestHTMLExtractor:
    """Tests for HTML text extraction."""

    def test_extract_simple_html(self):
        """Test extracting text from simple HTML."""
        html = """
        <html>
        <head><title>Test Page</title></head>
        <body>
            <h1>Hello World</h1>
            <p>This is a test paragraph.</p>
        </body>
        </html>
        """
        text, title = extract_text_from_html(html)

        assert title == "Test Page"
        assert "Hello World" in text
        assert "This is a test paragraph" in text

    def test_extract_removes_scripts(self):
        """Test that scripts are removed."""
        html = """
        <html>
        <body>
            <p>Content</p>
            <script>alert('evil');</script>
            <p>More content</p>
        </body>
        </html>
        """
        text, _ = extract_text_from_html(html)

        assert "Content" in text
        assert "More content" in text
        assert "alert" not in text
        assert "evil" not in text

    def test_extract_removes_nav_footer(self):
        """Test that nav and footer are removed."""
        html = """
        <html>
        <body>
            <nav>Navigation menu</nav>
            <main>Main content here</main>
            <footer>Footer text</footer>
        </body>
        </html>
        """
        text, _ = extract_text_from_html(html)

        assert "Main content" in text
        assert "Navigation menu" not in text
        assert "Footer text" not in text

    def test_extract_prefers_article(self):
        """Test that article content is preferred."""
        html = """
        <html>
        <body>
            <div class="sidebar">Sidebar content</div>
            <article>
                <h1>Article Title</h1>
                <p>Article body text.</p>
            </article>
        </body>
        </html>
        """
        text, _ = extract_text_from_html(html)

        assert "Article Title" in text
        assert "Article body" in text

    def test_extract_title_cleanup(self):
        """Test title cleanup removes site name."""
        html = """
        <html>
        <head><title>Article Title | Site Name</title></head>
        <body><p>Content</p></body>
        </html>
        """
        _, title = extract_text_from_html(html)

        assert title == "Article Title"

    def test_extract_article_content_with_metadata(self):
        """Test extracting article content with metadata."""
        html = """
        <html>
        <head>
            <title>Test Article</title>
            <meta property="og:title" content="OG Title">
            <meta name="author" content="John Doe">
            <meta name="description" content="A test article description">
        </head>
        <body>
            <article>Article body content here.</article>
        </body>
        </html>
        """
        content, metadata = extract_article_content(html)

        assert "Article body content" in content
        assert metadata.get("title") == "OG Title"
        assert metadata.get("author") == "John Doe"
        assert metadata.get("description") == "A test article description"


class TestPluginRegistry:
    """Tests for plugin registry with scrapers and PDF parsers."""

    def test_scraper_registration(self):
        """Test that HTTP scraper is registered."""
        from statement_extractor.pipeline.registry import PluginRegistry
        # Import to trigger registration
        from statement_extractor.plugins import scrapers  # noqa: F401

        scrapers_list = PluginRegistry.get_scrapers()
        assert len(scrapers_list) >= 1

        # Check HTTP scraper exists
        names = [s.name for s in scrapers_list]
        assert "http_scraper" in names

    def test_pdf_parser_registration(self):
        """Test that PDF parser is registered."""
        from statement_extractor.pipeline.registry import PluginRegistry
        # Import to trigger registration
        from statement_extractor.plugins import pdf  # noqa: F401

        parsers = PluginRegistry.get_pdf_parsers()
        assert len(parsers) >= 1

        # Check PyPDF parser exists
        names = [p.name for p in parsers]
        assert "pypdf_parser" in names


class TestHTTPScraperPlugin:
    """Tests for HTTP scraper plugin."""

    def test_content_type_detection_pdf_header(self):
        """Test PDF detection from content-type header."""
        from statement_extractor.plugins.scrapers.http import HttpScraperPlugin

        headers = {"content-type": "application/pdf"}
        content_type = HttpScraperPlugin._detect_content_type(headers, "https://example.com/doc")

        assert content_type == ContentType.PDF

    def test_content_type_detection_pdf_url(self):
        """Test PDF detection from URL extension."""
        from statement_extractor.plugins.scrapers.http import HttpScraperPlugin

        headers = {}
        content_type = HttpScraperPlugin._detect_content_type(
            headers, "https://example.com/report.pdf"
        )

        assert content_type == ContentType.PDF

    def test_content_type_detection_html(self):
        """Test HTML detection from content-type header."""
        from statement_extractor.plugins.scrapers.http import HttpScraperPlugin

        headers = {"content-type": "text/html; charset=utf-8"}
        content_type = HttpScraperPlugin._detect_content_type(headers, "https://example.com")

        assert content_type == ContentType.HTML

    def test_captcha_detection_cloudflare(self):
        """Test CAPTCHA detection for Cloudflare pages."""
        from statement_extractor.plugins.scrapers.http import HttpScraperPlugin

        content = b"<html><body>Checking your browser before accessing cloudflare</body></html>"
        assert HttpScraperPlugin._is_captcha_page(content) is True

    def test_captcha_detection_normal_page(self):
        """Test CAPTCHA detection for normal pages."""
        from statement_extractor.plugins.scrapers.http import HttpScraperPlugin

        content = b"<html><body>This is a normal article with lots of content. " * 100 + b"</body></html>"
        assert HttpScraperPlugin._is_captcha_page(content) is False


class TestURLLoaderConfig:
    """Tests for URLLoaderConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        from statement_extractor.document.loader import URLLoaderConfig

        config = URLLoaderConfig()

        assert config.timeout == 30.0
        assert config.use_ocr is False
        assert config.max_pdf_pages == 500
        assert config.scraper_plugin is None
        assert config.pdf_parser_plugin is None
        assert config.extract_metadata is True

    def test_custom_config(self):
        """Test custom configuration."""
        from statement_extractor.document.loader import URLLoaderConfig

        config = URLLoaderConfig(
            timeout=60.0,
            use_ocr=True,
            max_pdf_pages=100,
        )

        assert config.timeout == 60.0
        assert config.use_ocr is True
        assert config.max_pdf_pages == 100


@pytest.mark.asyncio
class TestURLLoaderAsync:
    """Async tests for URLLoader."""

    async def test_loader_initialization(self):
        """Test URLLoader initializes correctly."""
        from statement_extractor.document.loader import URLLoader, URLLoaderConfig

        config = URLLoaderConfig(timeout=10.0)
        loader = URLLoader(config)

        assert loader.config.timeout == 10.0

    async def test_loader_get_scraper(self):
        """Test URLLoader gets scraper plugin."""
        from statement_extractor.document.loader import URLLoader
        # Import to trigger registration
        from statement_extractor.plugins import scrapers  # noqa: F401

        loader = URLLoader()
        scraper = loader._get_scraper()

        assert scraper is not None
        assert scraper.name == "http_scraper"

    async def test_loader_get_pdf_parser(self):
        """Test URLLoader gets PDF parser plugin."""
        from statement_extractor.document.loader import URLLoader
        # Import to trigger registration
        from statement_extractor.plugins import pdf  # noqa: F401

        loader = URLLoader()
        parser = loader._get_pdf_parser()

        assert parser is not None
        assert parser.name == "pypdf_parser"


# Integration tests (require network, marked as slow)
@pytest.mark.slow
@pytest.mark.asyncio
class TestURLLoaderIntegration:
    """Integration tests for URL loading (requires network)."""

    async def test_load_html_page(self):
        """Test loading an HTML page."""
        from statement_extractor.document.loader import URLLoader

        loader = URLLoader()
        # Using example.com as a stable test target
        document = await loader.load("https://example.com")

        assert document is not None
        assert document.full_text
        assert "Example Domain" in document.metadata.title or "example" in document.full_text.lower()

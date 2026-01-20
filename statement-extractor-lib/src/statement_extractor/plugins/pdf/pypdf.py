"""
PDF parser plugin using PyMuPDF (fitz) with optional OCR fallback.

Extracts text from PDFs page by page, with automatic detection of
image-heavy PDFs that may require OCR.
"""

import io
import logging
import os
import tempfile
from typing import Any, Optional

from ..base import BasePDFParserPlugin, PDFParseResult
from ...pipeline.registry import PluginRegistry

logger = logging.getLogger(__name__)


@PluginRegistry.pdf_parser
class PyPDFParserPlugin(BasePDFParserPlugin):
    """
    PDF parser using PyMuPDF (fitz) with optional OCR fallback.

    Features:
    - Fast text extraction using PyMuPDF
    - Automatic detection of image-heavy PDFs
    - Optional OCR fallback using Tesseract
    - Metadata extraction (title, author, etc.)
    """

    def __init__(
        self,
        image_threshold: float = 0.5,
        text_threshold: float = 0.4,
        use_ocr_fallback: bool = True,
    ):
        """
        Initialize the PDF parser.

        Args:
            image_threshold: Images per page threshold for OCR trigger
            text_threshold: Text density threshold (chars/1000 per page)
            use_ocr_fallback: Enable automatic OCR for image-heavy PDFs
        """
        self._image_threshold = image_threshold
        self._text_threshold = text_threshold
        self._use_ocr_fallback = use_ocr_fallback

    @property
    def name(self) -> str:
        return "pypdf_parser"

    @property
    def priority(self) -> int:
        return 100

    @property
    def description(self) -> str:
        return "PDF parser using PyMuPDF with optional OCR fallback"

    @property
    def supports_ocr(self) -> bool:
        return self._use_ocr_fallback

    def parse(
        self,
        pdf_bytes: bytes,
        max_pages: int = 500,
        use_ocr: bool = False,
    ) -> PDFParseResult:
        """
        Extract text from PDF bytes.

        Args:
            pdf_bytes: Raw PDF file content
            max_pages: Maximum number of pages to process
            use_ocr: Force OCR even for text-extractable PDFs

        Returns:
            PDFParseResult with extracted text for each page
        """
        try:
            import fitz  # PyMuPDF
        except ImportError:
            return PDFParseResult(
                pages=[],
                page_count=0,
                error="PyMuPDF (fitz) not installed. Install with: pip install PyMuPDF",
            )

        temp_path: Optional[str] = None

        try:
            # Write bytes to temp file for fitz
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
                f.write(pdf_bytes)
                temp_path = f.name

            logger.info(f"Parsing PDF: {len(pdf_bytes)} bytes")

            # Open the PDF
            pdf_doc = fitz.open(temp_path)
            total_pages = len(pdf_doc)
            logger.info(f"PDF has {total_pages} pages")

            # Check if we should use OCR
            should_ocr = use_ocr or (
                self._use_ocr_fallback and self._is_mostly_images(pdf_doc)
            )

            if should_ocr:
                logger.info("PDF appears image-heavy, using OCR")
                result = self._parse_with_ocr(pdf_doc, max_pages)
            else:
                logger.info("PDF has extractable text, using direct extraction")
                result = self._parse_with_fitz(pdf_doc, max_pages)

            pdf_doc.close()
            return result

        except Exception as e:
            logger.exception(f"Error parsing PDF: {e}")
            return PDFParseResult(
                pages=[],
                page_count=0,
                error=f"Failed to parse PDF: {e}",
            )
        finally:
            # Clean up temp file
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except Exception:
                    pass

    def _is_mostly_images(self, pdf_doc) -> bool:
        """
        Check if PDF is mostly images (may need OCR).

        Args:
            pdf_doc: PyMuPDF document object

        Returns:
            True if PDF appears to be image-heavy
        """
        total_pages = len(pdf_doc)
        if total_pages == 0:
            return False

        # Count images in first few pages
        sample_pages = min(3, total_pages)
        image_count = 0
        for i in range(sample_pages):
            image_count += len(pdf_doc[i].get_images())

        avg_images_per_page = image_count / sample_pages

        # Check text density in sample pages
        sample_text = ""
        for i in range(sample_pages):
            sample_text += pdf_doc[i].get_text()

        text_density = len(sample_text) / 1000 / sample_pages

        logger.debug(
            f"PDF analysis: {avg_images_per_page:.1f} images/page, "
            f"{text_density:.2f} text density"
        )

        # If text density is high, don't use OCR
        if text_density > self._text_threshold:
            return False

        # If many images per page and low text, probably needs OCR
        return avg_images_per_page > self._image_threshold

    def _parse_with_fitz(self, pdf_doc, max_pages: int) -> PDFParseResult:
        """
        Extract text using PyMuPDF (fast, direct extraction).

        Args:
            pdf_doc: PyMuPDF document object
            max_pages: Maximum pages to process

        Returns:
            PDFParseResult with extracted text
        """
        pages = []
        total_pages = len(pdf_doc)

        for i in range(min(total_pages, max_pages)):
            page = pdf_doc[i]
            text = page.get_text()
            pages.append(text.strip())

            if (i + 1) % 50 == 0:
                logger.debug(f"Processed {i + 1}/{min(total_pages, max_pages)} pages")

        # Extract metadata
        metadata = self._extract_metadata(pdf_doc)

        return PDFParseResult(
            pages=pages,
            page_count=total_pages,
            metadata=metadata,
        )

    def _parse_with_ocr(self, pdf_doc, max_pages: int) -> PDFParseResult:
        """
        Extract text using OCR (Tesseract).

        Args:
            pdf_doc: PyMuPDF document object
            max_pages: Maximum pages to process

        Returns:
            PDFParseResult with OCR-extracted text
        """
        try:
            import pytesseract
            from PIL import Image
        except ImportError:
            return PDFParseResult(
                pages=[],
                page_count=len(pdf_doc),
                error="OCR dependencies not installed. Install with: pip install pytesseract Pillow",
            )

        pages = []
        total_pages = len(pdf_doc)

        for i in range(min(total_pages, max_pages)):
            page = pdf_doc[i]

            # Render page to image
            pix = page.get_pixmap(dpi=150)  # 150 DPI is good balance
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

            # Run OCR
            text = pytesseract.image_to_string(img)
            pages.append(text.strip())

            if (i + 1) % 10 == 0:
                logger.debug(f"OCR processed {i + 1}/{min(total_pages, max_pages)} pages")

        # Extract metadata
        metadata = self._extract_metadata(pdf_doc)

        return PDFParseResult(
            pages=pages,
            page_count=total_pages,
            metadata=metadata,
        )

    @staticmethod
    def _extract_metadata(pdf_doc) -> dict[str, Any]:
        """
        Extract PDF metadata.

        Args:
            pdf_doc: PyMuPDF document object

        Returns:
            Dictionary of metadata fields
        """
        metadata = {}

        try:
            doc_metadata = pdf_doc.metadata
            if doc_metadata:
                # Map common PDF metadata fields
                field_map = {
                    "title": "title",
                    "author": "author",
                    "subject": "subject",
                    "keywords": "keywords",
                    "creator": "creator",
                    "producer": "producer",
                    "creationDate": "created",
                    "modDate": "modified",
                }

                for pdf_key, our_key in field_map.items():
                    value = doc_metadata.get(pdf_key)
                    if value and isinstance(value, str) and value.strip():
                        metadata[our_key] = value.strip()
        except Exception as e:
            logger.debug(f"Error extracting metadata: {e}")

        return metadata

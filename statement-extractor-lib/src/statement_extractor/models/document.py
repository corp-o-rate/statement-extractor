"""
Document models for document-level processing.

Document: A document with metadata, pages, and optional summary
DocumentMetadata: Metadata about the document source
DocumentPage: A single page within a document
TextChunk: A chunk of text for processing with page tracking
ChunkingConfig: Configuration for text chunking
"""

import uuid
from typing import Any, Optional

from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    """
    Metadata about a document source.

    Contains information about where the document came from and
    who authored it, useful for generating citations.
    """
    url: Optional[str] = Field(None, description="URL source of the document")
    title: Optional[str] = Field(None, description="Document title")
    year: Optional[int] = Field(None, description="Publication year")
    authors: list[str] = Field(default_factory=list, description="List of authors")
    source_type: Optional[str] = Field(
        None,
        description="Type of source: 'pdf', 'webpage', 'text', etc."
    )
    custom: dict[str, Any] = Field(
        default_factory=dict,
        description="Custom metadata fields"
    )

    def format_citation(self, page_number: Optional[int] = None) -> str:
        """
        Format a citation string for this document.

        Args:
            page_number: Optional page number to include

        Returns:
            Citation string like "Title - Author, 2024, p. 5"
        """
        parts = []

        if self.title:
            parts.append(self.title)

        if self.authors:
            if len(self.authors) == 1:
                parts.append(self.authors[0])
            elif len(self.authors) == 2:
                parts.append(f"{self.authors[0]} & {self.authors[1]}")
            else:
                parts.append(f"{self.authors[0]} et al.")

        if self.year:
            parts.append(str(self.year))

        if page_number is not None:
            parts.append(f"p. {page_number}")

        return " - ".join(parts) if parts else ""


class DocumentPage(BaseModel):
    """
    A single page within a document.

    Tracks the page number and character offset for citation purposes.
    """
    page_number: int = Field(..., description="1-indexed page number")
    text: str = Field(..., description="Text content of the page")
    char_offset: int = Field(
        ...,
        description="Character offset of this page in the full document text"
    )

    @property
    def char_end(self) -> int:
        """Get the ending character offset of this page."""
        return self.char_offset + len(self.text)


class TextChunk(BaseModel):
    """
    A chunk of text for processing.

    Contains the text along with position tracking for mapping
    extracted statements back to their source pages.
    """
    chunk_index: int = Field(..., description="0-indexed chunk number")
    text: str = Field(..., description="Chunk text content")
    start_char: int = Field(..., description="Starting character offset in full document")
    end_char: int = Field(..., description="Ending character offset in full document")
    page_numbers: list[int] = Field(
        default_factory=list,
        description="Page numbers this chunk spans (1-indexed)"
    )
    token_count: int = Field(..., description="Number of tokens in this chunk")
    overlap_chars: int = Field(
        default=0,
        description="Number of characters of overlap from previous chunk"
    )
    document_id: str = Field(..., description="ID of the source document")

    @property
    def primary_page(self) -> Optional[int]:
        """Get the primary page number for this chunk (first page)."""
        return self.page_numbers[0] if self.page_numbers else None


class ChunkingConfig(BaseModel):
    """
    Configuration for document chunking.

    Controls how documents are split into chunks for processing.
    """
    max_tokens: int = Field(
        default=2000,
        ge=100,
        description="Maximum tokens per chunk (hard limit)"
    )
    target_tokens: int = Field(
        default=1000,
        ge=50,
        description="Target tokens per chunk (soft limit, prefers to split here)"
    )
    overlap_tokens: int = Field(
        default=100,
        ge=0,
        description="Tokens of overlap between consecutive chunks"
    )
    respect_page_boundaries: bool = Field(
        default=True,
        description="Try to split at page boundaries when possible"
    )
    respect_sentence_boundaries: bool = Field(
        default=True,
        description="Try to split at sentence boundaries when possible"
    )


class Document(BaseModel):
    """
    A document for processing through the extraction pipeline.

    Contains the full text, optional page structure, metadata for citations,
    and an optional summary for context.
    """
    document_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this document"
    )
    metadata: DocumentMetadata = Field(
        default_factory=DocumentMetadata,
        description="Document metadata for citations"
    )
    pages: list[DocumentPage] = Field(
        default_factory=list,
        description="List of pages (optional, for PDFs)"
    )
    full_text: str = Field(
        default="",
        description="Full text content of the document"
    )
    summary: Optional[str] = Field(
        None,
        description="Generated summary of the document"
    )

    @classmethod
    def from_text(
        cls,
        text: str,
        title: Optional[str] = None,
        url: Optional[str] = None,
        **metadata_kwargs,
    ) -> "Document":
        """
        Create a document from plain text.

        Args:
            text: The document text
            title: Optional document title
            url: Optional source URL
            **metadata_kwargs: Additional metadata fields

        Returns:
            Document instance
        """
        metadata = DocumentMetadata(
            title=title,
            url=url,
            source_type="text",
            **metadata_kwargs,
        )
        return cls(
            metadata=metadata,
            full_text=text,
        )

    @classmethod
    def from_pages(
        cls,
        pages: list[str],
        title: Optional[str] = None,
        source_type: str = "pdf",
        **metadata_kwargs,
    ) -> "Document":
        """
        Create a document from a list of page texts.

        Args:
            pages: List of page text strings (0-indexed input, stored as 1-indexed)
            title: Optional document title
            source_type: Source type (default: "pdf")
            **metadata_kwargs: Additional metadata fields

        Returns:
            Document instance
        """
        metadata = DocumentMetadata(
            title=title,
            source_type=source_type,
            **metadata_kwargs,
        )

        # Build pages with character offsets
        doc_pages = []
        char_offset = 0

        for i, page_text in enumerate(pages):
            doc_pages.append(DocumentPage(
                page_number=i + 1,  # 1-indexed
                text=page_text,
                char_offset=char_offset,
            ))
            char_offset += len(page_text)
            if i < len(pages) - 1:
                char_offset += 1  # Account for newline between pages

        # Join pages with newlines for full text
        full_text = "\n".join(pages)

        return cls(
            metadata=metadata,
            pages=doc_pages,
            full_text=full_text,
        )

    def get_page_at_char(self, char_offset: int) -> Optional[int]:
        """
        Get the page number containing a character offset.

        Args:
            char_offset: Character offset in full_text

        Returns:
            1-indexed page number, or None if no pages defined
        """
        if not self.pages:
            return None

        for page in self.pages:
            if page.char_offset <= char_offset < page.char_end:
                return page.page_number

        # If past the last page, return last page
        if char_offset >= self.pages[-1].char_end:
            return self.pages[-1].page_number

        return None

    def get_pages_in_range(self, start_char: int, end_char: int) -> list[int]:
        """
        Get all page numbers that overlap with a character range.

        Args:
            start_char: Start character offset
            end_char: End character offset

        Returns:
            List of 1-indexed page numbers
        """
        if not self.pages:
            return []

        page_numbers = []
        for page in self.pages:
            # Check if page overlaps with range
            if page.char_offset < end_char and page.char_end > start_char:
                page_numbers.append(page.page_number)

        return page_numbers

    @property
    def page_count(self) -> int:
        """Get the number of pages in the document."""
        return len(self.pages)

    @property
    def char_count(self) -> int:
        """Get the total character count."""
        return len(self.full_text)

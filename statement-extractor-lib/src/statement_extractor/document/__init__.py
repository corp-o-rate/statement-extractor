"""
Document processing module for statement extraction.

Provides document-level features including:
- Text chunking with page awareness
- Statement deduplication across chunks
- Document summarization
- Citation generation

Example:
    >>> from statement_extractor.document import DocumentPipeline, Document
    >>>
    >>> pipeline = DocumentPipeline()
    >>> document = Document.from_text("Your document text...", title="Report 2024")
    >>> ctx = pipeline.process(document)
    >>>
    >>> for stmt in ctx.labeled_statements:
    ...     print(f"{stmt.subject_fqn} -> {stmt.object_fqn}")
    ...     print(f"  Citation: {stmt.citation}")
"""

# Re-export document models for convenience
from ..models.document import (
    ChunkingConfig,
    Document,
    DocumentMetadata,
    DocumentPage,
    TextChunk,
)
from .chunker import DocumentChunker
from .context import DocumentContext
from .deduplicator import StatementDeduplicator
from .html_extractor import extract_text_from_html, extract_article_content
from .loader import URLLoader, URLLoaderConfig, load_url, load_url_sync
from .pipeline import DocumentPipeline, DocumentPipelineConfig
from .summarizer import DocumentSummarizer

__all__ = [
    # Pipeline
    "DocumentPipeline",
    "DocumentPipelineConfig",
    # Context
    "DocumentContext",
    # Components
    "DocumentChunker",
    "StatementDeduplicator",
    "DocumentSummarizer",
    # URL loading
    "URLLoader",
    "URLLoaderConfig",
    "load_url",
    "load_url_sync",
    # HTML extraction
    "extract_text_from_html",
    "extract_article_content",
    # Models (re-exported)
    "Document",
    "DocumentMetadata",
    "DocumentPage",
    "TextChunk",
    "ChunkingConfig",
]

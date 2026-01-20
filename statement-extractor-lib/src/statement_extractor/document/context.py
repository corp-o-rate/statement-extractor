"""
DocumentContext - Context object for document-level pipeline results.

Holds all results from document processing including chunks, statements,
and pipeline outputs.
"""

from typing import Any

from pydantic import BaseModel, Field

from ..models.document import Document, TextChunk
from ..models.labels import LabeledStatement
from ..models.statement import PipelineStatement, RawTriple
from ..pipeline.context import PipelineContext


class DocumentContext(BaseModel):
    """
    Context for document-level processing results.

    Contains the source document, chunks, and aggregated pipeline results.
    """
    document: Document = Field(..., description="Source document")
    chunks: list[TextChunk] = Field(
        default_factory=list,
        description="Text chunks created from the document"
    )

    # Aggregated pipeline results
    raw_triples: list[RawTriple] = Field(
        default_factory=list,
        description="Raw triples from all chunks (Stage 1)"
    )
    statements: list[PipelineStatement] = Field(
        default_factory=list,
        description="Pipeline statements from all chunks (Stage 2)"
    )
    labeled_statements: list[LabeledStatement] = Field(
        default_factory=list,
        description="Final labeled statements (Stage 5)"
    )

    # Processing metadata
    chunk_contexts: list[PipelineContext] = Field(
        default_factory=list,
        description="Individual pipeline contexts for each chunk"
    )
    stage_timings: dict[str, float] = Field(
        default_factory=dict,
        description="Total time spent in each stage across all chunks"
    )
    processing_errors: list[str] = Field(
        default_factory=list,
        description="Errors encountered during processing"
    )
    processing_warnings: list[str] = Field(
        default_factory=list,
        description="Warnings generated during processing"
    )

    # Deduplication stats
    pre_dedup_count: int = Field(
        default=0,
        description="Number of statements before deduplication"
    )
    post_dedup_count: int = Field(
        default=0,
        description="Number of statements after deduplication"
    )

    class Config:
        arbitrary_types_allowed = True  # Allow PipelineContext

    @property
    def statement_count(self) -> int:
        """Get the total number of final statements."""
        return len(self.labeled_statements)

    @property
    def chunk_count(self) -> int:
        """Get the number of chunks."""
        return len(self.chunks)

    @property
    def duplicates_removed(self) -> int:
        """Get the number of duplicate statements removed."""
        return self.pre_dedup_count - self.post_dedup_count

    def add_error(self, error: str) -> None:
        """Add a processing error."""
        self.processing_errors.append(error)

    def add_warning(self, warning: str) -> None:
        """Add a processing warning."""
        self.processing_warnings.append(warning)

    def record_timing(self, stage: str, duration: float) -> None:
        """
        Record timing for a stage (accumulates across chunks).

        Args:
            stage: Stage name
            duration: Duration in seconds
        """
        if stage in self.stage_timings:
            self.stage_timings[stage] += duration
        else:
            self.stage_timings[stage] = duration

    def merge_chunk_context(self, chunk_ctx: PipelineContext) -> None:
        """
        Merge results from a chunk's pipeline context.

        Args:
            chunk_ctx: Pipeline context from processing a chunk
        """
        self.chunk_contexts.append(chunk_ctx)

        # Merge timings
        for stage, duration in chunk_ctx.stage_timings.items():
            self.record_timing(stage, duration)

        # Merge errors and warnings
        self.processing_errors.extend(chunk_ctx.processing_errors)
        self.processing_warnings.extend(chunk_ctx.processing_warnings)

    def get_statements_by_page(self, page_number: int) -> list[LabeledStatement]:
        """
        Get all statements from a specific page.

        Args:
            page_number: 1-indexed page number

        Returns:
            List of statements from that page
        """
        return [
            stmt for stmt in self.labeled_statements
            if stmt.page_number == page_number
        ]

    def get_statements_by_chunk(self, chunk_index: int) -> list[LabeledStatement]:
        """
        Get all statements from a specific chunk.

        Args:
            chunk_index: 0-indexed chunk index

        Returns:
            List of statements from that chunk
        """
        return [
            stmt for stmt in self.labeled_statements
            if stmt.statement.chunk_index == chunk_index
        ]

    def as_dict(self) -> dict[str, Any]:
        """Convert to a dictionary representation."""
        return {
            "document_id": self.document.document_id,
            "document_title": self.document.metadata.title,
            "summary": self.document.summary,
            "chunk_count": self.chunk_count,
            "statement_count": self.statement_count,
            "duplicates_removed": self.duplicates_removed,
            "statements": [stmt.as_dict() for stmt in self.labeled_statements],
            "timings": self.stage_timings,
            "errors": self.processing_errors,
            "warnings": self.processing_warnings,
        }

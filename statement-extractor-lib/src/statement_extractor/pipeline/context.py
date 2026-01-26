"""
PipelineContext - Data container that flows through all pipeline stages.

The context accumulates outputs from each stage:
- Stage 1 (Splitting): split_sentences
- Stage 2 (Extraction): statements
- Stage 3 (Qualification): qualified_entities
- Stage 4 (Canonicalization): canonical_entities
- Stage 5 (Labeling): labeled_statements
"""

from typing import Any, Optional

from pydantic import BaseModel, Field

from ..models import (
    SplitSentence,
    PipelineStatement,
    QualifiedEntity,
    CanonicalEntity,
    LabeledStatement,
    TaxonomyResult,
)


class PipelineContext(BaseModel):
    """
    Context object that flows through all pipeline stages.

    Accumulates outputs from each stage and provides access to
    source text, metadata, and intermediate results.
    """
    # Input
    source_text: str = Field(..., description="Original input text")
    source_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata about the source (e.g., document ID, URL, timestamp)"
    )

    # Stage 1 output: Split sentences
    split_sentences: list[SplitSentence] = Field(
        default_factory=list,
        description="Atomic sentences from Stage 1 (Splitting)"
    )

    # Stage 2 output: Statements with extracted entities
    statements: list[PipelineStatement] = Field(
        default_factory=list,
        description="Statements from Stage 2 (Extraction)"
    )

    # Stage 3 output: Qualified entities (keyed by entity_ref)
    qualified_entities: dict[str, QualifiedEntity] = Field(
        default_factory=dict,
        description="Qualified entities from Stage 3 (Qualification)"
    )

    # Stage 4 output: Canonical entities (keyed by entity_ref)
    canonical_entities: dict[str, CanonicalEntity] = Field(
        default_factory=dict,
        description="Canonical entities from Stage 4 (Canonicalization)"
    )

    # Stage 5 output: Final labeled statements
    labeled_statements: list[LabeledStatement] = Field(
        default_factory=list,
        description="Final labeled statements from Stage 5 (Labeling)"
    )

    # Classification results from extractor (populated by GLiNER2 or similar)
    # Keyed by source_text -> label_type -> (label_value, confidence)
    classification_results: dict[str, dict[str, tuple[str, float]]] = Field(
        default_factory=dict,
        description="Pre-computed classification results from Stage 2 extractor"
    )

    # Stage 6 output: Taxonomy classifications
    # Keyed by (source_text, taxonomy_name) -> list of TaxonomyResult
    # Multiple labels may match a single statement above threshold
    taxonomy_results: dict[tuple[str, str], list[TaxonomyResult]] = Field(
        default_factory=dict,
        description="Taxonomy classifications from Stage 6 (multiple labels per statement)"
    )

    # Processing metadata
    processing_errors: list[str] = Field(
        default_factory=list,
        description="Errors encountered during processing"
    )
    processing_warnings: list[str] = Field(
        default_factory=list,
        description="Warnings generated during processing"
    )
    stage_timings: dict[str, float] = Field(
        default_factory=dict,
        description="Timing information for each stage (stage_name -> seconds)"
    )

    def add_error(self, error: str) -> None:
        """Add a processing error."""
        self.processing_errors.append(error)

    def add_warning(self, warning: str) -> None:
        """Add a processing warning."""
        self.processing_warnings.append(warning)

    def record_timing(self, stage: str, duration: float) -> None:
        """Record timing for a stage."""
        self.stage_timings[stage] = duration

    def get_entity_refs(self) -> set[str]:
        """Get all unique entity refs from statements."""
        refs = set()
        for stmt in self.statements:
            refs.add(stmt.subject.entity_ref)
            refs.add(stmt.object.entity_ref)
        return refs

    def get_qualified_entity(self, entity_ref: str) -> Optional[QualifiedEntity]:
        """Get qualified entity by ref, or None if not found."""
        return self.qualified_entities.get(entity_ref)

    def get_canonical_entity(self, entity_ref: str) -> Optional[CanonicalEntity]:
        """Get canonical entity by ref, or None if not found."""
        return self.canonical_entities.get(entity_ref)

    def get_classification(
        self,
        source_text: str,
        label_type: str,
    ) -> Optional[tuple[str, float]]:
        """
        Get pre-computed classification result for a source text.

        Args:
            source_text: The source text that was classified
            label_type: The type of label (e.g., "sentiment")

        Returns:
            Tuple of (label_value, confidence) or None if not found
        """
        if source_text in self.classification_results:
            return self.classification_results[source_text].get(label_type)
        return None

    def set_classification(
        self,
        source_text: str,
        label_type: str,
        label_value: str,
        confidence: float,
    ) -> None:
        """
        Store a classification result for a source text.

        Args:
            source_text: The source text that was classified
            label_type: The type of label (e.g., "sentiment")
            label_value: The classification result (e.g., "positive")
            confidence: Confidence score (0.0 to 1.0)
        """
        if source_text not in self.classification_results:
            self.classification_results[source_text] = {}
        self.classification_results[source_text][label_type] = (label_value, confidence)

    @property
    def has_errors(self) -> bool:
        """Check if any errors occurred during processing."""
        return len(self.processing_errors) > 0

    @property
    def statement_count(self) -> int:
        """Get the number of statements in the final output."""
        return len(self.labeled_statements) if self.labeled_statements else len(self.statements)

    class Config:
        arbitrary_types_allowed = True

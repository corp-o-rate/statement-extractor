"""
Label models for the extraction pipeline.

StatementLabel: A label applied to a statement
LabeledStatement: Final output from Stage 5 with all labels
TaxonomyResult: Taxonomy classification from Stage 6
"""

from typing import Any, Optional, Union

from pydantic import BaseModel, Field

from .statement import PipelineStatement
from .canonical import CanonicalEntity


class StatementLabel(BaseModel):
    """
    A label applied to a statement in Stage 5 (Labeling).

    Labels can represent sentiment, relation type, confidence, or
    any other classification applied by labeler plugins.
    """
    label_type: str = Field(
        ...,
        description="Type of label: 'sentiment', 'relation_type', 'confidence', etc."
    )
    label_value: Union[str, float, bool] = Field(
        ...,
        description="The label value (string for classification, float for scores)"
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence in this label"
    )
    labeler: Optional[str] = Field(
        None,
        description="Name of the labeler plugin that produced this label"
    )

    def is_high_confidence(self, threshold: float = 0.8) -> bool:
        """Check if this is a high-confidence label."""
        return self.confidence >= threshold


class LabeledStatement(BaseModel):
    """
    Final output from Stage 5 (Labeling) with taxonomy from Stage 6.

    Contains the original statement, canonicalized subject and object,
    all labels applied by labeler plugins, and taxonomy classifications.
    """
    statement: PipelineStatement = Field(
        ...,
        description="The original statement from Stage 2"
    )
    subject_canonical: CanonicalEntity = Field(
        ...,
        description="Canonicalized subject entity"
    )
    object_canonical: CanonicalEntity = Field(
        ...,
        description="Canonicalized object entity"
    )
    labels: list[StatementLabel] = Field(
        default_factory=list,
        description="Labels applied to this statement"
    )
    taxonomy_results: list["TaxonomyResult"] = Field(
        default_factory=list,
        description="Taxonomy classifications from Stage 6"
    )
    # Document tracking fields
    document_id: Optional[str] = Field(
        None,
        description="ID of the source document (for document pipeline)"
    )
    page_number: Optional[int] = Field(
        None,
        description="Page number where this statement was extracted (1-indexed)"
    )
    citation: Optional[str] = Field(
        None,
        description="Formatted citation string (e.g., 'Title - Author, 2024, p. 5')"
    )

    def get_label(self, label_type: str) -> Optional[StatementLabel]:
        """Get a label by type, or None if not found."""
        for label in self.labels:
            if label.label_type == label_type:
                return label
        return None

    def get_labels_by_type(self, label_type: str) -> list[StatementLabel]:
        """Get all labels of a specific type."""
        return [label for label in self.labels if label.label_type == label_type]

    def add_label(self, label: StatementLabel) -> None:
        """Add a label to this statement."""
        self.labels.append(label)

    @property
    def subject_fqn(self) -> str:
        """Get the subject's fully qualified name."""
        return self.subject_canonical.fqn

    @property
    def object_fqn(self) -> str:
        """Get the object's fully qualified name."""
        return self.object_canonical.fqn

    def __str__(self) -> str:
        """Format as FQN triple."""
        return f"{self.subject_fqn} --[{self.statement.predicate}]--> {self.object_fqn}"

    def _build_entity_dict(self, canonical: CanonicalEntity, entity_type: str) -> dict:
        """Build entity dict for serialization."""
        statement_entity = self.statement.subject if entity_type == "subject" else self.statement.object
        fqn = self.subject_fqn if entity_type == "subject" else self.object_fqn

        # Get canonical_id from identifiers or canonical_match
        identifiers = canonical.qualified_entity.qualifiers.identifiers
        canonical_id = identifiers.get("canonical_id")
        if not canonical_id and canonical.canonical_match:
            canonical_id = canonical.canonical_match.canonical_id

        result = {
            "text": statement_entity.text,
            "type": statement_entity.type.value,
            "fqn": fqn,
            "canonical_id": canonical_id,
        }

        # Add name if available
        if canonical.name:
            result["name"] = canonical.name

        # Add qualifiers if available
        qualifiers_dict = canonical.qualifiers_dict
        if qualifiers_dict:
            result["qualifiers"] = qualifiers_dict

        return result

    def as_dict(self) -> dict:
        """Convert to a simplified dictionary representation."""
        return {
            "subject": self._build_entity_dict(self.subject_canonical, "subject"),
            "predicate": self.statement.predicate,
            "object": self._build_entity_dict(self.object_canonical, "object"),
            "source_text": self.statement.source_text,
            "labels": {
                label.label_type: label.label_value
                for label in self.labels
            },
            "taxonomy": [
                {
                    "category": t.category,
                    "label": t.label,
                    "confidence": t.confidence,
                }
                for t in self.taxonomy_results
            ],
            "document_id": self.document_id,
            "page_number": self.page_number,
            "citation": self.citation,
        }

    class Config:
        frozen = False  # Allow modification during pipeline stages


class TaxonomyResult(BaseModel):
    """
    Result of taxonomy classification from Stage 6.

    Represents a classification of a statement against a taxonomy,
    typically with a category (top-level) and label (specific topic).
    """
    taxonomy_name: str = Field(
        ...,
        description="Name of the taxonomy (e.g., 'esg_topics', 'industry_codes')"
    )
    category: str = Field(
        ...,
        description="Top-level category (e.g., 'environment', 'governance')"
    )
    label: str = Field(
        ...,
        description="Specific label within the category (e.g., 'carbon emissions')"
    )
    label_id: Optional[int] = Field(
        None,
        description="Numeric ID for reproducibility"
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Classification confidence"
    )
    classifier: Optional[str] = Field(
        None,
        description="Name of the taxonomy plugin that produced this result"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (e.g., runner-up labels, scores)"
    )

    @property
    def full_label(self) -> str:
        """Get the full label in category:label format."""
        return f"{self.category}:{self.label}"

    def is_high_confidence(self, threshold: float = 0.7) -> bool:
        """Check if this is a high-confidence classification."""
        return self.confidence >= threshold

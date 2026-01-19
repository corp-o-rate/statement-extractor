"""
Label models for the extraction pipeline.

StatementLabel: A label applied to a statement
LabeledStatement: Final output from Stage 5 with all labels
"""

from typing import Optional, Union

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
    Final output from Stage 5 (Labeling).

    Contains the original statement, canonicalized subject and object,
    and all labels applied by labeler plugins.
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

    def as_dict(self) -> dict:
        """Convert to a simplified dictionary representation."""
        return {
            "subject": {
                "text": self.statement.subject.text,
                "type": self.statement.subject.type.value,
                "fqn": self.subject_fqn,
                "canonical_id": (
                    self.subject_canonical.canonical_match.canonical_id
                    if self.subject_canonical.canonical_match else None
                ),
            },
            "predicate": self.statement.predicate,
            "object": {
                "text": self.statement.object.text,
                "type": self.statement.object.type.value,
                "fqn": self.object_fqn,
                "canonical_id": (
                    self.object_canonical.canonical_match.canonical_id
                    if self.object_canonical.canonical_match else None
                ),
            },
            "source_text": self.statement.source_text,
            "labels": {
                label.label_type: label.label_value
                for label in self.labels
            },
        }

    class Config:
        frozen = False  # Allow modification during pipeline stages

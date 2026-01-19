"""
ConfidenceLabeler - Aggregates confidence scores from all pipeline stages.

Combines:
- Statement extraction confidence
- Entity extraction confidence
- Canonical match confidence
"""

import logging
from typing import Optional

from ..base import BaseLabelerPlugin, PluginCapability
from ...pipeline.context import PipelineContext
from ...pipeline.registry import PluginRegistry
from ...models import (
    PipelineStatement,
    CanonicalEntity,
    StatementLabel,
)

logger = logging.getLogger(__name__)


@PluginRegistry.labeler
class ConfidenceLabeler(BaseLabelerPlugin):
    """
    Labeler that aggregates confidence scores from all pipeline stages.

    Produces an overall confidence score for each statement.
    """

    def __init__(
        self,
        statement_weight: float = 0.4,
        subject_weight: float = 0.2,
        object_weight: float = 0.2,
        canonical_weight: float = 0.2,
    ):
        """
        Initialize the confidence labeler.

        Args:
            statement_weight: Weight for statement extraction confidence
            subject_weight: Weight for subject entity confidence
            object_weight: Weight for object entity confidence
            canonical_weight: Weight for canonical match confidence
        """
        self._statement_weight = statement_weight
        self._subject_weight = subject_weight
        self._object_weight = object_weight
        self._canonical_weight = canonical_weight

    @property
    def name(self) -> str:
        return "confidence_labeler"

    @property
    def priority(self) -> int:
        return 100  # Run after other labelers

    @property
    def capabilities(self) -> PluginCapability:
        return PluginCapability.NONE

    @property
    def description(self) -> str:
        return "Aggregates confidence scores from all pipeline stages"

    @property
    def label_type(self) -> str:
        return "confidence"

    def label(
        self,
        statement: PipelineStatement,
        subject_canonical: CanonicalEntity,
        object_canonical: CanonicalEntity,
        context: PipelineContext,
    ) -> Optional[StatementLabel]:
        """
        Calculate aggregate confidence for a statement.

        Args:
            statement: The statement to label
            subject_canonical: Canonicalized subject
            object_canonical: Canonicalized object
            context: Pipeline context

        Returns:
            StatementLabel with aggregate confidence
        """
        scores = []
        weights = []

        # Statement confidence
        if statement.confidence_score is not None:
            scores.append(statement.confidence_score)
            weights.append(self._statement_weight)

        # Subject entity confidence
        scores.append(statement.subject.confidence)
        weights.append(self._subject_weight)

        # Object entity confidence
        scores.append(statement.object.confidence)
        weights.append(self._object_weight)

        # Canonical match confidence
        subj_canon_conf = (
            subject_canonical.canonical_match.match_confidence
            if subject_canonical.canonical_match else 0.5
        )
        obj_canon_conf = (
            object_canonical.canonical_match.match_confidence
            if object_canonical.canonical_match else 0.5
        )
        avg_canon_conf = (subj_canon_conf + obj_canon_conf) / 2
        scores.append(avg_canon_conf)
        weights.append(self._canonical_weight)

        # Calculate weighted average
        total_weight = sum(weights)
        if total_weight > 0:
            aggregate_confidence = sum(s * w for s, w in zip(scores, weights)) / total_weight
        else:
            aggregate_confidence = 0.5

        return StatementLabel(
            label_type=self.label_type,
            label_value=round(aggregate_confidence, 3),
            confidence=1.0,  # High confidence in our calculation
            labeler=self.name,
        )


# Allow importing without decorator for testing
ConfidenceLabelerClass = ConfidenceLabeler

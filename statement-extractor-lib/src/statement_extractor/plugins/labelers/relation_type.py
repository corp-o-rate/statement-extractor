"""
RelationTypeLabeler - Uses predicate category from GLiNER2 extraction.

The relation type comes from the predicate category assigned during
Stage 2 extraction (GLiNER2). If no category is available, logs an error.
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
class RelationTypeLabeler(BaseLabelerPlugin):
    """
    Labeler that uses predicate category from GLiNER2 as relation type.

    The category is set during Stage 2 extraction when GLiNER2 matches
    a predicate from default_predicates.json (organized by category).
    """

    @property
    def name(self) -> str:
        return "relation_type_labeler"

    @property
    def priority(self) -> int:
        return 20

    @property
    def capabilities(self) -> PluginCapability:
        return PluginCapability.NONE

    @property
    def description(self) -> str:
        return "Uses predicate category from GLiNER2 as relation type"

    @property
    def label_type(self) -> str:
        return "relation_type"

    def label(
        self,
        statement: PipelineStatement,
        subject_canonical: CanonicalEntity,
        object_canonical: CanonicalEntity,
        context: PipelineContext,
    ) -> Optional[StatementLabel]:
        """
        Get relation type from statement's predicate category.

        Args:
            statement: The statement to label
            subject_canonical: Canonicalized subject
            object_canonical: Canonicalized object
            context: Pipeline context

        Returns:
            StatementLabel with relation type, or None if no category
        """
        if not statement.predicate_category:
            logger.error(
                f"No predicate_category for statement: "
                f"'{statement.subject.text}' --[{statement.predicate}]--> '{statement.object.text}'"
            )
            return None

        return StatementLabel(
            label_type=self.label_type,
            label_value=statement.predicate_category,
            confidence=statement.confidence_score,  # Use statement's confidence
            labeler=self.name,
        )


# Allow importing without decorator for testing
RelationTypeLabelerClass = RelationTypeLabeler

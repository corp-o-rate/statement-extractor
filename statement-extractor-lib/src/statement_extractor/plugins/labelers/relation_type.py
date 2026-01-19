"""
RelationTypeLabeler - Maps predicates to canonical relation types.

Uses pattern matching to classify predicates into a standard taxonomy.
"""

import logging
import re
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

# Relation type taxonomy with patterns
RELATION_TAXONOMY = {
    "employment": [
        r'\b(works?\s+for|employed\s+by|works?\s+at|joined|hired|appointed|CEO|CFO|CTO|COO|president|director|manager)\b',
        r'\b(chief\s+\w+\s+officer|vice\s+president|VP|head\s+of)\b',
    ],
    "acquisition": [
        r'\b(acquired|bought|purchased|took\s+over|merged\s+with)\b',
    ],
    "partnership": [
        r'\b(partnered|collaborated|allied|joined\s+forces|teamed\s+up)\b',
    ],
    "investment": [
        r'\b(invested|funded|backed|raised|financing|funding\s+round)\b',
    ],
    "product_launch": [
        r'\b(launched|released|unveiled|announced|introduced|rolled\s+out)\b',
    ],
    "legal": [
        r'\b(sued|filed\s+lawsuit|accused|charged|indicted|settled)\b',
    ],
    "location": [
        r'\b(located|based|headquartered|operates\s+in|expanded\s+to)\b',
    ],
    "ownership": [
        r'\b(owns|owned|subsidiary|parent\s+company|controls)\b',
    ],
    "founding": [
        r'\b(founded|started|established|created|co-founded)\b',
    ],
    "competition": [
        r'\b(competes|competing|rival|competitor|vs\.?)\b',
    ],
    "supply": [
        r'\b(supplies|provides|delivers|supplies\s+to|vendor|supplier)\b',
    ],
    "customer": [
        r'\b(customer|client|buys\s+from|purchases\s+from|uses)\b',
    ],
}


def classify_relation(predicate: str, source_text: str) -> tuple[str, float]:
    """
    Classify a predicate into a relation type.

    Returns (relation_type, confidence).
    """
    combined = f"{predicate} {source_text}".lower()

    best_type = "other"
    best_score = 0

    for rel_type, patterns in RELATION_TAXONOMY.items():
        score = 0
        for pattern in patterns:
            matches = len(re.findall(pattern, combined, re.IGNORECASE))
            score += matches

        if score > best_score:
            best_score = score
            best_type = rel_type

    # Calculate confidence based on score
    if best_score == 0:
        return "other", 0.5
    elif best_score == 1:
        return best_type, 0.7
    elif best_score == 2:
        return best_type, 0.8
    else:
        return best_type, 0.9


@PluginRegistry.labeler
class RelationTypeLabeler(BaseLabelerPlugin):
    """
    Labeler that classifies predicates into relation types.

    Maps predicates to a standard taxonomy (employment, acquisition, etc.).
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
        return "Classifies predicates into relation type taxonomy"

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
        Classify relation type of a statement.

        Args:
            statement: The statement to label
            subject_canonical: Canonicalized subject
            object_canonical: Canonicalized object
            context: Pipeline context

        Returns:
            StatementLabel with relation type
        """
        relation_type, confidence = classify_relation(
            statement.predicate,
            statement.source_text,
        )

        return StatementLabel(
            label_type=self.label_type,
            label_value=relation_type,
            confidence=confidence,
            labeler=self.name,
        )


# Allow importing without decorator for testing
RelationTypeLabelerClass = RelationTypeLabeler

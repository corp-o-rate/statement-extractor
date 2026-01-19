"""
SentimentLabeler - Classifies statement sentiment.

Uses GLiNER2 classification when available, falls back to pattern matching.
"""

import logging
import re
from typing import Optional

from ..base import BaseLabelerPlugin, ClassificationSchema, PluginCapability
from ...pipeline.context import PipelineContext
from ...pipeline.registry import PluginRegistry
from ...models import (
    PipelineStatement,
    CanonicalEntity,
    StatementLabel,
)

logger = logging.getLogger(__name__)

# Positive predicates and patterns
POSITIVE_PATTERNS = [
    r'\b(acquired|announced|launched|released|expanded|grew|increased|improved|won|awarded)\b',
    r'\b(partnered|collaborated|joined|signed|agreed|approved|completed|achieved)\b',
    r'\b(invested|funded|raised|promoted|hired|appointed)\b',
]

# Negative predicates and patterns
NEGATIVE_PATTERNS = [
    r'\b(lost|declined|decreased|dropped|fell|failed|fired|laid off|resigned)\b',
    r'\b(sued|accused|charged|investigated|fined|penalized|rejected|denied)\b',
    r'\b(closed|shut down|cancelled|terminated|withdrew|abandoned)\b',
]

# Neutral predicates (don't include words that are also in positive/negative lists)
NEUTRAL_PATTERNS = [
    r'\b(said|stated|reported|confirmed|disclosed)\b',
    r'\b(is|was|are|were|has|have|had)\b',
    r'\b(located|based|headquartered|operates|employs)\b',
]


def classify_sentiment(text: str) -> tuple[str, float]:
    """
    Classify sentiment of text using pattern matching.

    Returns (sentiment, confidence) where sentiment is 'positive', 'negative', or 'neutral'.
    """
    text_lower = text.lower()

    positive_matches = sum(
        len(re.findall(pattern, text_lower, re.IGNORECASE))
        for pattern in POSITIVE_PATTERNS
    )
    negative_matches = sum(
        len(re.findall(pattern, text_lower, re.IGNORECASE))
        for pattern in NEGATIVE_PATTERNS
    )
    neutral_matches = sum(
        len(re.findall(pattern, text_lower, re.IGNORECASE))
        for pattern in NEUTRAL_PATTERNS
    )

    total_matches = positive_matches + negative_matches + neutral_matches

    if total_matches == 0:
        return "neutral", 0.5

    if positive_matches > negative_matches and positive_matches > neutral_matches:
        confidence = min(0.6 + (positive_matches / total_matches) * 0.3, 0.9)
        return "positive", confidence

    if negative_matches > positive_matches and negative_matches > neutral_matches:
        confidence = min(0.6 + (negative_matches / total_matches) * 0.3, 0.9)
        return "negative", confidence

    return "neutral", 0.6


@PluginRegistry.labeler
class SentimentLabeler(BaseLabelerPlugin):
    """
    Labeler that classifies statement sentiment.

    Provides a ClassificationSchema so GLiNER2 can run classification.
    Falls back to pattern matching if no pre-computed result available.
    """

    SENTIMENT_CHOICES = ["positive", "negative", "neutral"]

    @property
    def name(self) -> str:
        return "sentiment_labeler"

    @property
    def priority(self) -> int:
        return 10

    @property
    def capabilities(self) -> PluginCapability:
        return PluginCapability.NONE

    @property
    def description(self) -> str:
        return "Classifies statement sentiment (positive/negative/neutral)"

    @property
    def label_type(self) -> str:
        return "sentiment"

    @property
    def classification_schema(self) -> ClassificationSchema:
        """Provide schema for GLiNER2 to run classification."""
        return ClassificationSchema(
            label_type=self.label_type,
            choices=self.SENTIMENT_CHOICES,
            description="Classify the sentiment of this statement",
            scope="statement",
        )

    def label(
        self,
        statement: PipelineStatement,
        subject_canonical: CanonicalEntity,
        object_canonical: CanonicalEntity,
        context: PipelineContext,
    ) -> Optional[StatementLabel]:
        """
        Classify sentiment of a statement.

        First checks for pre-computed result from extractor (GLiNER2).
        Falls back to pattern matching if not available.
        """
        # Check for pre-computed classification from extractor
        result = context.get_classification(statement.source_text, self.label_type)
        if result:
            label_value, confidence = result
            return StatementLabel(
                label_type=self.label_type,
                label_value=label_value,
                confidence=confidence,
                labeler=self.name,
            )

        # Fallback: pattern-based classification
        text_to_analyze = f"{statement.predicate} {statement.source_text}"
        sentiment, confidence = classify_sentiment(text_to_analyze)

        return StatementLabel(
            label_type=self.label_type,
            label_value=sentiment,
            confidence=confidence,
            labeler=self.name,
        )


# Allow importing without decorator for testing
SentimentLabelerClass = SentimentLabeler

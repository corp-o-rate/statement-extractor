"""
Labeler plugins for Stage 5 (Labeling).

Applies labels to statements (sentiment, relation type, confidence).
"""

from .base import BaseLabelerPlugin
from .sentiment import SentimentLabeler
from .relation_type import RelationTypeLabeler
from .confidence import ConfidenceLabeler

__all__ = [
    "BaseLabelerPlugin",
    "SentimentLabeler",
    "RelationTypeLabeler",
    "ConfidenceLabeler",
]

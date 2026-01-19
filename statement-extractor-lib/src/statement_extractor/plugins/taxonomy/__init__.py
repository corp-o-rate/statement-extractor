"""
Taxonomy classifier plugins for Stage 6 (Taxonomy).

Classifies statements against large taxonomies using MNLI or embeddings.
"""

from .mnli import MNLITaxonomyClassifier
from .embedding import EmbeddingTaxonomyClassifier

__all__ = [
    "MNLITaxonomyClassifier",
    "EmbeddingTaxonomyClassifier",
]

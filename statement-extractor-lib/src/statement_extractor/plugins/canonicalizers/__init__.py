"""
Canonicalizer plugins for Stage 4 (Canonicalization).

Resolves entities to their canonical forms.
"""

from .base import BaseCanonicalizerPlugin
from .organization import OrganizationCanonicalizer
from .person import PersonCanonicalizer
from .location import LocationCanonicalizer

__all__ = [
    "BaseCanonicalizerPlugin",
    "OrganizationCanonicalizer",
    "PersonCanonicalizer",
    "LocationCanonicalizer",
]

"""
Extractor plugins for Stage 2 (Extraction).

Refines raw triples into statements with typed entities.
"""

from .base import BaseExtractorPlugin
from .gliner2 import GLiNER2Extractor

__all__ = [
    "BaseExtractorPlugin",
    "GLiNER2Extractor",
]

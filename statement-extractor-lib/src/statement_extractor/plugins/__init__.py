"""
Plugins module for the extraction pipeline.

Contains all plugin implementations organized by stage:
- splitters/: Stage 1 - Text to atomic triples
- extractors/: Stage 2 - Refine entities and relations
- qualifiers/: Stage 3 - Add qualifiers and identifiers
- canonicalizers/: Stage 4 - Resolve canonical forms
- labelers/: Stage 5 - Classify statements
- taxonomy/: Stage 6 - Taxonomy classification
"""

from .base import (
    PluginCapability,
    BasePlugin,
    BaseSplitterPlugin,
    BaseExtractorPlugin,
    BaseQualifierPlugin,
    BaseCanonicalizerPlugin,
    BaseLabelerPlugin,
    BaseTaxonomyPlugin,
)

# Import plugin modules for auto-registration
from . import splitters, extractors, qualifiers, canonicalizers, labelers, taxonomy

__all__ = [
    "PluginCapability",
    "BasePlugin",
    "BaseSplitterPlugin",
    "BaseExtractorPlugin",
    "BaseQualifierPlugin",
    "BaseCanonicalizerPlugin",
    "BaseLabelerPlugin",
    "BaseTaxonomyPlugin",
    # Plugin modules
    "splitters",
    "extractors",
    "qualifiers",
    "canonicalizers",
    "labelers",
    "taxonomy",
]

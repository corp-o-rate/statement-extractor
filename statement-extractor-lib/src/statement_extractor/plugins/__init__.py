"""
Plugins module for the extraction pipeline.

Contains all plugin implementations organized by stage:
- splitters/: Stage 1 - Text to atomic triples
- extractors/: Stage 2 - Refine entities and relations
- qualifiers/: Stage 3 - Qualify entities (add identifiers, canonical names, FQN)
- labelers/: Stage 4 - Classify statements
- taxonomy/: Stage 5 - Taxonomy classification
"""

from .base import (
    PluginCapability,
    BasePlugin,
    BaseSplitterPlugin,
    BaseExtractorPlugin,
    BaseQualifierPlugin,
    BaseLabelerPlugin,
    BaseTaxonomyPlugin,
    # Content acquisition plugins
    ContentType,
    ScraperResult,
    PDFParseResult,
    BaseScraperPlugin,
    BasePDFParserPlugin,
)

# Import plugin modules for auto-registration
from . import splitters, extractors, qualifiers, labelers, taxonomy
# Content acquisition plugins
from . import scrapers, pdf

__all__ = [
    "PluginCapability",
    "BasePlugin",
    "BaseSplitterPlugin",
    "BaseExtractorPlugin",
    "BaseQualifierPlugin",
    "BaseLabelerPlugin",
    "BaseTaxonomyPlugin",
    # Content acquisition plugins
    "ContentType",
    "ScraperResult",
    "PDFParseResult",
    "BaseScraperPlugin",
    "BasePDFParserPlugin",
    # Plugin modules
    "splitters",
    "extractors",
    "qualifiers",
    "labelers",
    "taxonomy",
    "scrapers",
    "pdf",
]

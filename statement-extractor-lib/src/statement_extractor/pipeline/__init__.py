"""
Pipeline module for the extraction pipeline.

This module provides the core pipeline infrastructure:
- PipelineContext: Data container that flows through all stages
- PipelineConfig: Configuration for stage/plugin selection
- PluginRegistry: Registration and discovery of plugins
- ExtractionPipeline: Main orchestrator class

Plugins are auto-loaded when this module is imported.
"""

from .context import PipelineContext
from .config import PipelineConfig
from .registry import PluginRegistry
from .orchestrator import ExtractionPipeline


def _load_plugins():
    """Load all plugins by importing their modules."""
    import logging

    try:
        from ..plugins import splitters, extractors, qualifiers, canonicalizers, labelers, taxonomy
        # The @PluginRegistry decorators register plugins on import
    except ImportError as e:
        logging.debug(f"Some plugins failed to load: {e}")


# Auto-load plugins on module import
_load_plugins()


__all__ = [
    "PipelineContext",
    "PipelineConfig",
    "PluginRegistry",
    "ExtractionPipeline",
]

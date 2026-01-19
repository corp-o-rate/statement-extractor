"""
Pipeline module for the extraction pipeline.

This module provides the core pipeline infrastructure:
- PipelineContext: Data container that flows through all stages
- PipelineConfig: Configuration for stage/plugin selection
- PluginRegistry: Registration and discovery of plugins
- ExtractionPipeline: Main orchestrator class
"""

from .context import PipelineContext
from .config import PipelineConfig
from .registry import PluginRegistry
from .orchestrator import ExtractionPipeline

__all__ = [
    "PipelineContext",
    "PipelineConfig",
    "PluginRegistry",
    "ExtractionPipeline",
]

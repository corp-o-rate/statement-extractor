"""
Splitter plugins for Stage 1 (Splitting).

Splits text into atomic triples.
"""

from .base import BaseSplitterPlugin
from .t5_gemma import T5GemmaSplitter

__all__ = [
    "BaseSplitterPlugin",
    "T5GemmaSplitter",
]

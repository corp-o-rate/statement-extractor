"""
Statement Extractor - Extract structured statements from text using T5-Gemma 2.

A Python library for extracting subject-predicate-object triples from unstructured text.
Uses Diverse Beam Search (Vijayakumar et al., 2016) for high-quality extraction.

Paper: https://arxiv.org/abs/1610.02424

Example:
    >>> from statement_extractor import extract_statements
    >>> result = extract_statements("Apple Inc. announced a new iPhone today.")
    >>> for stmt in result:
    ...     print(f"{stmt.subject.text} -> {stmt.predicate} -> {stmt.object.text}")
    Apple Inc. -> announced -> a new iPhone

    >>> # Get as different formats
    >>> xml = extract_statements_as_xml("Some text...")
    >>> json_str = extract_statements_as_json("Some text...")
    >>> data = extract_statements_as_dict("Some text...")
"""

__version__ = "0.1.0"

from .models import (
    Entity,
    EntityType,
    ExtractionOptions,
    ExtractionResult,
    Statement,
)

from .extractor import (
    StatementExtractor,
    extract_statements,
    extract_statements_as_dict,
    extract_statements_as_json,
    extract_statements_as_xml,
)

__all__ = [
    # Version
    "__version__",
    # Models
    "Entity",
    "EntityType",
    "ExtractionOptions",
    "ExtractionResult",
    "Statement",
    # Extractor class
    "StatementExtractor",
    # Convenience functions
    "extract_statements",
    "extract_statements_as_dict",
    "extract_statements_as_json",
    "extract_statements_as_xml",
]
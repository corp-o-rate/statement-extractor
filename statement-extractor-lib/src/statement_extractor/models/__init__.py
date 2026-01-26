"""
Data models for the extraction pipeline.

This module contains all Pydantic models used throughout the pipeline stages:
- Stage 1 (Splitting): RawTriple
- Stage 2 (Extraction): ExtractedEntity, PipelineStatement
- Stage 3 (Qualification): EntityQualifiers, QualifiedEntity
- Stage 4 (Canonicalization): CanonicalMatch, CanonicalEntity
- Stage 5 (Labeling): StatementLabel, LabeledStatement

It also re-exports all models from the original models.py for backward compatibility.
"""

# Import from the original models.py file (now a sibling at the same level)
# We need to import these BEFORE the local modules to avoid circular imports
import sys
import importlib.util
from pathlib import Path

# Manually load the old models.py to avoid conflict with this package
_models_py_path = Path(__file__).parent.parent / "models.py"
if _models_py_path.exists():
    _spec = importlib.util.spec_from_file_location("_old_models", _models_py_path)
    _old_models = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_old_models)

    # Re-export everything from the old models
    Entity = _old_models.Entity
    ExtractionMethod = _old_models.ExtractionMethod
    Statement = _old_models.Statement
    ExtractionResult = _old_models.ExtractionResult
    PredicateMatch = _old_models.PredicateMatch
    PredicateTaxonomy = _old_models.PredicateTaxonomy
    PredicateComparisonConfig = _old_models.PredicateComparisonConfig
    ScoringConfig = _old_models.ScoringConfig
    ExtractionOptions = _old_models.ExtractionOptions

    # Use EntityType from old models
    EntityType = _old_models.EntityType
else:
    # Fallback: define locally if old models.py doesn't exist
    from .entity import EntityType

# New pipeline models
from .entity import ExtractedEntity
from .statement import SplitSentence, RawTriple, PipelineStatement
from .qualifiers import EntityQualifiers, QualifiedEntity, ResolvedRole, ResolvedOrganization
from .canonical import CanonicalMatch, CanonicalEntity
from .labels import StatementLabel, LabeledStatement, TaxonomyResult
from .document import (
    Document,
    DocumentMetadata,
    DocumentPage,
    TextChunk,
    ChunkingConfig,
)

__all__ = [
    # Re-exported from original models.py (backward compatibility)
    "Entity",
    "EntityType",
    "ExtractionMethod",
    "Statement",
    "ExtractionResult",
    "PredicateMatch",
    "PredicateTaxonomy",
    "PredicateComparisonConfig",
    "ScoringConfig",
    "ExtractionOptions",
    # New pipeline models
    "ExtractedEntity",
    "SplitSentence",
    "RawTriple",  # Backwards compatibility alias for SplitSentence
    "PipelineStatement",
    "EntityQualifiers",
    "QualifiedEntity",
    "ResolvedRole",
    "ResolvedOrganization",
    "CanonicalMatch",
    "CanonicalEntity",
    "StatementLabel",
    "LabeledStatement",
    "TaxonomyResult",
    # Document models
    "Document",
    "DocumentMetadata",
    "DocumentPage",
    "TextChunk",
    "ChunkingConfig",
]

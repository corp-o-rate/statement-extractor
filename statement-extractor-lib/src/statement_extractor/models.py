"""Pydantic models for statement extraction results."""

from enum import Enum
from pathlib import Path
from typing import Callable, Optional

from pydantic import BaseModel, Field


class EntityType(str, Enum):
    """Supported entity types for subjects and objects."""
    ORG = "ORG"
    PERSON = "PERSON"
    GPE = "GPE"  # Geopolitical entity (countries, cities, states)
    LOC = "LOC"  # Non-GPE locations
    PRODUCT = "PRODUCT"
    EVENT = "EVENT"
    WORK_OF_ART = "WORK_OF_ART"
    LAW = "LAW"
    DATE = "DATE"
    MONEY = "MONEY"
    PERCENT = "PERCENT"
    QUANTITY = "QUANTITY"
    UNKNOWN = "UNKNOWN"


class ExtractionMethod(str, Enum):
    """Method used to extract the triple components."""
    HYBRID = "hybrid"  # Model subject/object + GLiNER2 predicate
    GLINER = "gliner"  # All components from GLiNER2 extraction
    MODEL = "model"  # All components from T5-Gemma model (when GLiNER2 disabled)


class Entity(BaseModel):
    """An entity (subject or object) with its text and type."""
    text: str = Field(..., description="The entity text")
    type: EntityType = Field(default=EntityType.UNKNOWN, description="The entity type")

    def __str__(self) -> str:
        return f"{self.text} ({self.type.value})"

    def merge_type_from(self, other: "Entity") -> "Entity":
        """
        Return a new Entity with the more specific type.

        If this entity has UNKNOWN type and other has a specific type,
        returns a new entity with this text but other's type.
        Otherwise returns self unchanged.
        """
        if self.type == EntityType.UNKNOWN and other.type != EntityType.UNKNOWN:
            return Entity(text=self.text, type=other.type)
        return self


class Statement(BaseModel):
    """A single extracted statement (subject-predicate-object triple)."""
    subject: Entity = Field(..., description="The subject entity")
    predicate: str = Field(..., description="The relationship/predicate")
    object: Entity = Field(..., description="The object entity")
    source_text: Optional[str] = Field(None, description="The original text this statement was extracted from")

    # Extraction method tracking
    extraction_method: ExtractionMethod = Field(
        default=ExtractionMethod.MODEL,
        description="Method used to extract this triple (hybrid, spacy, split, or model)"
    )

    # Quality scoring fields
    confidence_score: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Semantic similarity score (0-1) between source text and reassembled triple"
    )
    evidence_span: Optional[tuple[int, int]] = Field(
        None,
        description="Character offsets (start, end) in source text where this triple is grounded"
    )
    canonical_predicate: Optional[str] = Field(
        None,
        description="Canonical form of the predicate if taxonomy matching was used"
    )
    was_reversed: bool = Field(
        default=False,
        description="True if subject/object were swapped during reversal detection"
    )

    def __str__(self) -> str:
        return f"{self.subject.text} -- {self.predicate} --> {self.object.text}"

    def as_triple(self) -> tuple[str, str, str]:
        """Return as a simple (subject, predicate, object) tuple."""
        return (self.subject.text, self.predicate, self.object.text)

    def merge_entity_types_from(self, other: "Statement") -> "Statement":
        """
        Return a new Statement with more specific entity types merged from other.

        If this statement has UNKNOWN entity types and other has specific types,
        the returned statement will use the specific types from other.
        All other fields come from self.
        """
        merged_subject = self.subject.merge_type_from(other.subject)
        merged_object = self.object.merge_type_from(other.object)

        # Only create new statement if something changed
        if merged_subject is self.subject and merged_object is self.object:
            return self

        return Statement(
            subject=merged_subject,
            object=merged_object,
            predicate=self.predicate,
            source_text=self.source_text,
            extraction_method=self.extraction_method,
            confidence_score=self.confidence_score,
            evidence_span=self.evidence_span,
            canonical_predicate=self.canonical_predicate,
            was_reversed=self.was_reversed,
        )

    def reversed(self) -> "Statement":
        """
        Return a new Statement with subject and object swapped.

        Sets was_reversed=True to indicate the swap occurred.
        """
        return Statement(
            subject=self.object,
            object=self.subject,
            predicate=self.predicate,
            source_text=self.source_text,
            extraction_method=self.extraction_method,
            confidence_score=self.confidence_score,
            evidence_span=self.evidence_span,
            canonical_predicate=self.canonical_predicate,
            was_reversed=True,
        )


class ExtractionResult(BaseModel):
    """The result of statement extraction from text."""
    statements: list[Statement] = Field(default_factory=list, description="List of extracted statements")
    source_text: Optional[str] = Field(None, description="The original input text")

    def __len__(self) -> int:
        return len(self.statements)

    def __iter__(self):
        return iter(self.statements)

    def to_triples(self) -> list[tuple[str, str, str]]:
        """Return all statements as simple (subject, predicate, object) tuples."""
        return [stmt.as_triple() for stmt in self.statements]


# =============================================================================
# Predicate Taxonomy & Comparison Configuration
# =============================================================================

class PredicateMatch(BaseModel):
    """Result of matching a predicate to a canonical form."""
    original: str = Field(..., description="The original extracted predicate")
    canonical: Optional[str] = Field(None, description="Matched canonical predicate, if any")
    similarity: float = Field(default=0.0, ge=0.0, le=1.0, description="Cosine similarity score")
    matched: bool = Field(default=False, description="Whether a canonical match was found above threshold")


class PredicateTaxonomy(BaseModel):
    """A taxonomy of canonical predicates for normalization."""
    predicates: list[str] = Field(..., description="List of canonical predicate forms")
    name: Optional[str] = Field(None, description="Optional taxonomy name for identification")

    @classmethod
    def from_file(cls, path: str | Path) -> "PredicateTaxonomy":
        """Load taxonomy from a file (one predicate per line)."""
        with open(path, "r") as f:
            predicates = [line.strip() for line in f if line.strip() and not line.startswith("#")]
        return cls(predicates=predicates)

    @classmethod
    def from_list(cls, predicates: list[str], name: Optional[str] = None) -> "PredicateTaxonomy":
        """Create taxonomy from a list of predicates."""
        return cls(predicates=predicates, name=name)


class PredicateComparisonConfig(BaseModel):
    """Configuration for embedding-based predicate comparison."""
    embedding_model: str = Field(
        default="sentence-transformers/paraphrase-MiniLM-L6-v2",
        description="Sentence-transformers model ID for computing embeddings"
    )
    similarity_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum cosine similarity to consider a taxonomy match"
    )
    dedup_threshold: float = Field(
        default=0.65,
        ge=0.0,
        le=1.0,
        description="Minimum similarity to consider predicates duplicates"
    )
    normalize_text: bool = Field(
        default=True,
        description="Lowercase and strip predicates before embedding"
    )


# =============================================================================
# Scoring Configuration
# =============================================================================

class ScoringConfig(BaseModel):
    """Configuration for beam scoring and triple quality assessment."""
    quality_weight: float = Field(
        default=1.0,
        ge=0.0,
        description="Weight for confidence scores in beam selection"
    )
    coverage_weight: float = Field(
        default=0.5,
        ge=0.0,
        description="Weight (β) for coverage bonus - how much source text is explained"
    )
    redundancy_penalty: float = Field(
        default=0.3,
        ge=0.0,
        description="Penalty (γ) for redundant/near-duplicate triples"
    )
    length_penalty: float = Field(
        default=0.1,
        ge=0.0,
        description="Penalty (δ) for verbosity - discourages overly long outputs"
    )
    min_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum confidence score to keep a triple (0=recall, 0.5=balanced, 0.8=precision)"
    )
    merge_top_n: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of top beams to merge when merge_beams=True"
    )


# =============================================================================
# Extraction Options
# =============================================================================

class ExtractionOptions(BaseModel):
    """Options for controlling the extraction process."""

    # Beam search parameters
    num_beams: int = Field(default=4, ge=1, le=16, description="Number of beams for diverse beam search")
    diversity_penalty: float = Field(default=1.0, ge=0.0, description="Penalty for beam diversity")
    max_new_tokens: int = Field(default=2048, ge=128, le=8192, description="Maximum tokens to generate")
    min_statement_ratio: float = Field(default=1.0, ge=0.0, description="Minimum statements per sentence ratio")
    max_attempts: int = Field(default=3, ge=1, le=10, description="Maximum extraction retry attempts")
    deduplicate: bool = Field(default=True, description="Remove duplicate statements")

    # Predicate taxonomy & comparison
    predicate_taxonomy: Optional[PredicateTaxonomy] = Field(
        None,
        description="Optional canonical predicate taxonomy for normalization"
    )
    predicate_config: Optional[PredicateComparisonConfig] = Field(
        None,
        description="Configuration for predicate comparison (uses defaults if not provided)"
    )

    # Scoring configuration
    scoring_config: Optional[ScoringConfig] = Field(
        None,
        description="Configuration for quality scoring and beam selection"
    )

    # Pluggable canonicalization function
    entity_canonicalizer: Optional[Callable[[str], str]] = Field(
        None,
        description="Custom function to canonicalize entity text for deduplication"
    )

    # Mode flags (defaults favor quality)
    merge_beams: bool = Field(
        default=True,
        description="Merge top-N beams instead of selecting single best beam"
    )
    embedding_dedup: bool = Field(
        default=True,
        description="Use embedding similarity for predicate deduplication"
    )
    use_gliner_extraction: bool = Field(
        default=True,
        description="Use GLiNER2 for predicate/subject/object extraction (model provides structure + coreference)"
    )

    # GLiNER2 predicate configuration
    predicates: Optional[list[str]] = Field(
        default=None,
        description="Optional list of predefined predicate types for GLiNER2 relation extraction (e.g., ['works_for', 'founded'])"
    )
    use_default_predicates: bool = Field(
        default=True,
        description="Use default predicate taxonomy when no custom predicates provided (enables GLiNER2 relation extraction)"
    )

    # Verbose logging
    verbose: bool = Field(
        default=False,
        description="Enable verbose logging for debugging"
    )

    # Triple selection
    all_triples: bool = Field(
        default=False,
        description="Keep all candidate triples instead of selecting the highest-scoring one per source"
    )

    class Config:
        arbitrary_types_allowed = True  # Allow Callable type

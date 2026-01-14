"""Pydantic models for statement extraction results."""

from enum import Enum
from typing import Optional
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


class Entity(BaseModel):
    """An entity (subject or object) with its text and type."""
    text: str = Field(..., description="The entity text")
    type: EntityType = Field(default=EntityType.UNKNOWN, description="The entity type")

    def __str__(self) -> str:
        return f"{self.text} ({self.type.value})"


class Statement(BaseModel):
    """A single extracted statement (subject-predicate-object triple)."""
    subject: Entity = Field(..., description="The subject entity")
    predicate: str = Field(..., description="The relationship/predicate")
    object: Entity = Field(..., description="The object entity")
    source_text: Optional[str] = Field(None, description="The original text this statement was extracted from")

    def __str__(self) -> str:
        return f"{self.subject.text} -- {self.predicate} --> {self.object.text}"

    def as_triple(self) -> tuple[str, str, str]:
        """Return as a simple (subject, predicate, object) tuple."""
        return (self.subject.text, self.predicate, self.object.text)


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


class ExtractionOptions(BaseModel):
    """Options for controlling the extraction process."""
    num_beams: int = Field(default=4, ge=1, le=16, description="Number of beams for diverse beam search")
    diversity_penalty: float = Field(default=1.0, ge=0.0, description="Penalty for beam diversity")
    max_new_tokens: int = Field(default=2048, ge=128, le=8192, description="Maximum tokens to generate")
    min_statement_ratio: float = Field(default=1.0, ge=0.0, description="Minimum statements per sentence ratio")
    max_attempts: int = Field(default=3, ge=1, le=10, description="Maximum extraction retry attempts")
    deduplicate: bool = Field(default=True, description="Remove duplicate statements")
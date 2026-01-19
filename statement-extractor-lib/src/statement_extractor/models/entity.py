"""
Entity models for the extraction pipeline.

ExtractedEntity represents entities identified during extraction with
confidence scores and span information.

Note: EntityType is imported from the original models.py for consistency.
"""

from typing import Optional, TYPE_CHECKING
import uuid

from pydantic import BaseModel, Field

# Import EntityType from parent module to avoid duplication
# This will be populated by __init__.py which loads from old models.py
if TYPE_CHECKING:
    from enum import Enum

    class EntityType(str, Enum):
        """Supported entity types for subjects and objects."""
        ORG = "ORG"
        PERSON = "PERSON"
        GPE = "GPE"
        LOC = "LOC"
        PRODUCT = "PRODUCT"
        EVENT = "EVENT"
        WORK_OF_ART = "WORK_OF_ART"
        LAW = "LAW"
        DATE = "DATE"
        MONEY = "MONEY"
        PERCENT = "PERCENT"
        QUANTITY = "QUANTITY"
        UNKNOWN = "UNKNOWN"
else:
    # At runtime, we need to import it from somewhere
    # Try the old models.py location first
    try:
        import importlib.util
        from pathlib import Path
        _models_py_path = Path(__file__).parent.parent / "models.py"
        _spec = importlib.util.spec_from_file_location("_old_models", _models_py_path)
        _old_models = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_old_models)
        EntityType = _old_models.EntityType
    except Exception:
        # Fallback to defining it here
        from enum import Enum

        class EntityType(str, Enum):
            """Supported entity types for subjects and objects."""
            ORG = "ORG"
            PERSON = "PERSON"
            GPE = "GPE"
            LOC = "LOC"
            PRODUCT = "PRODUCT"
            EVENT = "EVENT"
            WORK_OF_ART = "WORK_OF_ART"
            LAW = "LAW"
            DATE = "DATE"
            MONEY = "MONEY"
            PERCENT = "PERCENT"
            QUANTITY = "QUANTITY"
            UNKNOWN = "UNKNOWN"


class ExtractedEntity(BaseModel):
    """
    An entity extracted from text with type and confidence information.

    Used in Stage 2 (Extraction) and flows through subsequent stages.
    """
    text: str = Field(..., description="The entity text as extracted")
    type: EntityType = Field(default=EntityType.UNKNOWN, description="The entity type")
    span: Optional[tuple[int, int]] = Field(
        None,
        description="Character offsets (start, end) in source text"
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence score for this entity extraction"
    )
    entity_ref: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique reference ID for tracking this entity through the pipeline"
    )

    def __str__(self) -> str:
        return f"{self.text} ({self.type.value})"

    def __hash__(self) -> int:
        return hash(self.entity_ref)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ExtractedEntity):
            return False
        return self.entity_ref == other.entity_ref

    class Config:
        frozen = False  # Allow modification during pipeline stages

"""
Qualifier models for the extraction pipeline.

EntityQualifiers: Semantic qualifiers and external identifiers
QualifiedEntity: Entity with qualification information from Stage 3
"""

from typing import Optional

from pydantic import BaseModel, Field

from .entity import EntityType


class EntityQualifiers(BaseModel):
    """
    Qualifiers that provide context and identifiers for an entity.

    Populated by Stage 3 (Qualification) plugins such as:
    - PersonQualifierPlugin: Adds role, org for PERSON entities
    - GLEIFQualifierPlugin: Adds LEI for ORG entities
    - CompaniesHouseQualifierPlugin: Adds UK company number
    - SECEdgarQualifierPlugin: Adds SEC CIK, ticker
    """
    # Canonical name from database (for ORG entities)
    legal_name: Optional[str] = Field(None, description="Canonical legal name from database")

    # Semantic qualifiers (for PERSON entities)
    org: Optional[str] = Field(None, description="Organization/employer name")
    role: Optional[str] = Field(None, description="Job title/position/role")

    # Location qualifiers
    region: Optional[str] = Field(None, description="State/province/region")
    country: Optional[str] = Field(None, description="Country name or ISO code")
    city: Optional[str] = Field(None, description="City name")
    jurisdiction: Optional[str] = Field(None, description="Legal jurisdiction (e.g., 'UK', 'US-DE')")

    # External identifiers (keyed by identifier type)
    identifiers: dict[str, str] = Field(
        default_factory=dict,
        description="External identifiers: lei, ch_number, sec_cik, ticker, wikidata_qid, etc."
    )

    def has_any_qualifier(self) -> bool:
        """Check if any qualifier or identifier is set."""
        return bool(
            self.legal_name or self.org or self.role or self.region or self.country or
            self.city or self.jurisdiction or self.identifiers
        )

    def merge_with(self, other: "EntityQualifiers") -> "EntityQualifiers":
        """
        Merge qualifiers from another instance, preferring non-None values.

        Returns a new EntityQualifiers with merged values.
        """
        merged_identifiers = {**self.identifiers, **other.identifiers}
        return EntityQualifiers(
            legal_name=other.legal_name or self.legal_name,
            org=other.org or self.org,
            role=other.role or self.role,
            region=other.region or self.region,
            country=other.country or self.country,
            city=other.city or self.city,
            jurisdiction=other.jurisdiction or self.jurisdiction,
            identifiers=merged_identifiers,
        )


class QualifiedEntity(BaseModel):
    """
    An entity with qualification information from Stage 3.

    Links back to the original ExtractedEntity via entity_ref and
    adds qualifiers from various qualification plugins.
    """
    entity_ref: str = Field(..., description="Reference to the original ExtractedEntity")
    original_text: str = Field(..., description="Original entity text")
    entity_type: EntityType = Field(..., description="Entity type")
    qualifiers: EntityQualifiers = Field(
        default_factory=EntityQualifiers,
        description="Qualifiers and identifiers for this entity"
    )
    qualification_sources: list[str] = Field(
        default_factory=list,
        description="List of plugins that contributed qualifiers"
    )

    def add_qualifier_source(self, source: str) -> None:
        """Add a qualification source to the list."""
        if source not in self.qualification_sources:
            self.qualification_sources.append(source)

    class Config:
        frozen = False  # Allow modification during pipeline stages

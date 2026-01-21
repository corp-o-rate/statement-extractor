"""
Canonical models for the extraction pipeline.

CanonicalMatch: Result of matching to a canonical form
CanonicalEntity: Entity with canonical form from Stage 4
"""

from typing import Optional

from pydantic import BaseModel, Field

from .qualifiers import QualifiedEntity


class CanonicalMatch(BaseModel):
    """
    Result of matching an entity to its canonical form in Stage 4.

    Contains information about how the match was made and confidence level.
    """
    canonical_id: Optional[str] = Field(
        None,
        description="ID in canonical database (e.g., LEI, Wikidata QID)"
    )
    canonical_name: Optional[str] = Field(
        None,
        description="Canonical name/label"
    )
    match_method: str = Field(
        ...,
        description="How the match was made: 'identifier', 'name_exact', 'name_fuzzy', 'llm_verified'"
    )
    match_confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence in the canonical match"
    )
    match_details: Optional[dict] = Field(
        None,
        description="Additional details about the match (e.g., fuzzy score, LLM reasoning)"
    )

    def is_high_confidence(self, threshold: float = 0.85) -> bool:
        """Check if this is a high-confidence match."""
        return self.match_confidence >= threshold


class CanonicalEntity(BaseModel):
    """
    An entity with canonical form from Stage 4 (Canonicalization).

    Contains the qualified entity plus its canonical match (if found)
    and a fully qualified name (FQN) for display.
    """
    entity_ref: str = Field(..., description="Reference to the original ExtractedEntity")
    qualified_entity: QualifiedEntity = Field(
        ...,
        description="The qualified entity from Stage 3"
    )
    canonical_match: Optional[CanonicalMatch] = Field(
        None,
        description="Canonical match if found"
    )
    fqn: str = Field(
        ...,
        description="Fully qualified name, e.g., 'AMAZON CORP INC (SEC-CIK,USA)'"
    )

    @property
    def name(self) -> Optional[str]:
        """Get the canonical/legal name if available."""
        # Prefer legal_name from qualifiers (set by embedding qualifier)
        if self.qualified_entity.qualifiers.legal_name:
            return self.qualified_entity.qualifiers.legal_name
        # Fall back to canonical match name
        if self.canonical_match and self.canonical_match.canonical_name:
            return self.canonical_match.canonical_name
        return None

    @property
    def qualifiers_dict(self) -> Optional[dict[str, str]]:
        """
        Get qualifiers as a dict for serialization.

        Returns a dict with keys like: legal_name, region, source, source_id
        Only returns non-None values. Returns None if no qualifiers are set.
        """
        qualifiers = self.qualified_entity.qualifiers
        identifiers = qualifiers.identifiers
        result = {}

        # Add legal name
        if qualifiers.legal_name:
            result["legal_name"] = qualifiers.legal_name

        # Add region (prefer region, fall back to jurisdiction/country)
        if qualifiers.region:
            result["region"] = qualifiers.region
        elif qualifiers.jurisdiction:
            result["region"] = qualifiers.jurisdiction
        elif qualifiers.country:
            result["region"] = qualifiers.country

        # Add source and source_id from identifiers
        if "source" in identifiers:
            result["source"] = identifiers["source"]
        if "source_id" in identifiers:
            result["source_id"] = identifiers["source_id"]

        return result if result else None

    @classmethod
    def from_qualified(
        cls,
        qualified: QualifiedEntity,
        canonical_match: Optional[CanonicalMatch] = None,
        fqn: Optional[str] = None,
    ) -> "CanonicalEntity":
        """Create a CanonicalEntity from a QualifiedEntity."""
        if fqn is None:
            # Generate default FQN from qualifiers
            fqn = cls._generate_fqn(qualified, canonical_match)

        return cls(
            entity_ref=qualified.entity_ref,
            qualified_entity=qualified,
            canonical_match=canonical_match,
            fqn=fqn,
        )

    @staticmethod
    def _generate_fqn(
        qualified: QualifiedEntity,
        canonical_match: Optional[CanonicalMatch] = None
    ) -> str:
        """
        Generate a fully qualified name from qualifiers.

        Examples:
        - PERSON with role+org: "Tim Cook (CEO, Apple Inc)"
        - ORG with canonical: "Apple Inc (AAPL)"
        - PERSON with no qualifiers: "Tim Cook"
        """
        # Use canonical name if available, otherwise fall back to original text
        if canonical_match and canonical_match.canonical_name:
            base_name = canonical_match.canonical_name
        else:
            base_name = qualified.original_text

        qualifiers = qualified.qualifiers
        parts = []
        seen = set()  # Track seen values to avoid duplicates

        def add_part(value: str) -> None:
            """Add a part if not already seen (case-insensitive)."""
            if value and value.lower() not in seen:
                parts.append(value)
                seen.add(value.lower())

        # Add role for PERSON entities
        if qualifiers.role:
            add_part(qualifiers.role)

        # Add organization for PERSON entities
        if qualifiers.org:
            add_part(qualifiers.org)

        # Add ticker for ORG entities
        if "ticker" in qualifiers.identifiers:
            add_part(qualifiers.identifiers["ticker"])

        # Add jurisdiction if relevant
        if qualifiers.jurisdiction and not qualifiers.org:
            add_part(qualifiers.jurisdiction)

        if parts:
            return f"{base_name} ({', '.join(parts)})"
        return base_name

    class Config:
        frozen = False  # Allow modification during pipeline stages

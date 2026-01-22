"""
Pydantic models for organization/entity database records.
"""

from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


SourceType = Literal["gleif", "sec_edgar", "companies_house", "wikipedia"]


class EntityType(str, Enum):
    """
    Classification of organization type.

    Used to distinguish between businesses, non-profits, government agencies, etc.
    """
    # Business entities
    BUSINESS = "business"  # General business/company
    FUND = "fund"  # Investment funds, ETFs, mutual funds
    BRANCH = "branch"  # Branch offices of companies

    # Non-profit/civil society
    NONPROFIT = "nonprofit"  # Non-profit organizations
    NGO = "ngo"  # Non-governmental organizations
    FOUNDATION = "foundation"  # Charitable foundations
    TRADE_UNION = "trade_union"  # Labor unions

    # Government/public sector
    GOVERNMENT = "government"  # Government agencies
    INTERNATIONAL_ORG = "international_org"  # UN, WHO, IMF, etc.
    POLITICAL_PARTY = "political_party"  # Political parties

    # Education/research
    EDUCATIONAL = "educational"  # Schools, universities
    RESEARCH = "research"  # Research institutes

    # Other organization types
    RELIGIOUS = "religious"  # Religious organizations
    SPORTS = "sports"  # Sports clubs/teams
    MEDIA = "media"  # Media companies, studios
    HEALTHCARE = "healthcare"  # Hospitals, healthcare orgs

    # Unknown/unclassified
    UNKNOWN = "unknown"  # Type not determined


class CompanyRecord(BaseModel):
    """
    An organization record for the embedding database.

    Used for storing and searching organizations by embedding similarity.
    Note: Class name kept as CompanyRecord for API compatibility.
    """
    name: str = Field(..., description="Organization name (used for embedding and display)")
    source: SourceType = Field(..., description="Data source")
    source_id: str = Field(..., description="Unique identifier from source (LEI, CIK, CH number)")
    region: str = Field(default="", description="Geographic region/country (e.g., 'UK', 'US', 'DE')")
    entity_type: EntityType = Field(default=EntityType.UNKNOWN, description="Organization type classification")
    record: dict[str, Any] = Field(default_factory=dict, description="Original record from source")

    @property
    def canonical_id(self) -> str:
        """Generate canonical ID in format source:source_id."""
        return f"{self.source}:{self.source_id}"

    def model_dump_for_db(self) -> dict[str, Any]:
        """Convert to dict suitable for database storage."""
        return {
            "name": self.name,
            "source": self.source,
            "source_id": self.source_id,
            "region": self.region,
            "entity_type": self.entity_type.value,
            "record": self.record,
        }


class CompanyMatch(BaseModel):
    """
    An organization match result from embedding search.

    Returned by the organization qualifier when finding potential matches.
    Note: Class name kept as CompanyMatch for API compatibility.
    """
    query_name: str = Field(..., description="Name extracted from text (the search query)")
    record: CompanyRecord = Field(..., description="The matched organization record")
    source: SourceType = Field(..., description="Data source of match")
    source_id: str = Field(..., description="Source identifier of match")
    canonical_id: str = Field(..., description="Canonical ID in format source:source_id")
    similarity_score: float = Field(..., description="Embedding similarity score (0-1)")
    llm_confirmed: bool = Field(default=False, description="Whether LLM confirmed this match")

    @property
    def name(self) -> str:
        """Get the matched organization name."""
        return self.record.name

    @classmethod
    def from_record(
        cls,
        query_name: str,
        record: CompanyRecord,
        similarity_score: float,
        llm_confirmed: bool = False,
    ) -> "CompanyMatch":
        """Create a CompanyMatch from an organization record."""
        return cls(
            query_name=query_name,
            record=record,
            source=record.source,
            source_id=record.source_id,
            canonical_id=record.canonical_id,
            similarity_score=similarity_score,
            llm_confirmed=llm_confirmed,
        )


class DatabaseStats(BaseModel):
    """Statistics about the organization database."""
    total_records: int = 0
    by_source: dict[str, int] = Field(default_factory=dict)
    embedding_dimension: int = 0
    database_size_bytes: int = 0

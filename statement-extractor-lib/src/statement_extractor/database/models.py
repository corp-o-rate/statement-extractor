"""
Pydantic models for company database records.
"""

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


SourceType = Literal["gleif", "sec_edgar", "companies_house", "wikipedia"]


class CompanyRecord(BaseModel):
    """
    A company record for the embedding database.

    Used for storing and searching companies by embedding similarity.
    """
    name: str = Field(..., description="Searchable company name (used for embedding)")
    embedding_name: str = Field(..., description="Name used for embedding (usually same as name)")
    legal_name: str = Field(..., description="Official legal name")
    source: SourceType = Field(..., description="Data source")
    source_id: str = Field(..., description="Unique identifier from source (LEI, CIK, CH number)")
    region: str = Field(default="", description="Geographic region/country (e.g., 'UK', 'US', 'DE')")
    record: dict[str, Any] = Field(default_factory=dict, description="Original record from source")

    @property
    def canonical_id(self) -> str:
        """Generate canonical ID in format source:source_id."""
        return f"{self.source}:{self.source_id}"

    def model_dump_for_db(self) -> dict[str, Any]:
        """Convert to dict suitable for database storage."""
        return {
            "name": self.name,
            "embedding_name": self.embedding_name,
            "legal_name": self.legal_name,
            "source": self.source,
            "source_id": self.source_id,
            "region": self.region,
            "record": self.record,
        }


class CompanyMatch(BaseModel):
    """
    A company match result from embedding search.

    Returned by the company qualifier when finding potential matches.
    """
    name: str = Field(..., description="Name extracted from text")
    record: CompanyRecord = Field(..., description="The matched company record")
    legal_name: str = Field(..., description="Legal name from matched record")
    source: SourceType = Field(..., description="Data source of match")
    source_id: str = Field(..., description="Source identifier of match")
    canonical_id: str = Field(..., description="Canonical ID in format source:source_id")
    similarity_score: float = Field(..., description="Embedding similarity score (0-1)")
    llm_confirmed: bool = Field(default=False, description="Whether LLM confirmed this match")

    @classmethod
    def from_record(
        cls,
        query_name: str,
        record: CompanyRecord,
        similarity_score: float,
        llm_confirmed: bool = False,
    ) -> "CompanyMatch":
        """Create a CompanyMatch from a CompanyRecord."""
        return cls(
            name=query_name,
            record=record,
            legal_name=record.legal_name,
            source=record.source,
            source_id=record.source_id,
            canonical_id=record.canonical_id,
            similarity_score=similarity_score,
            llm_confirmed=llm_confirmed,
        )


class DatabaseStats(BaseModel):
    """Statistics about the company database."""
    total_records: int = 0
    by_source: dict[str, int] = Field(default_factory=dict)
    embedding_dimension: int = 0
    database_size_bytes: int = 0

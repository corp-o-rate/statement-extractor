"""
Entity/Organization database module for embedding-based entity qualification.

Provides:
- CompanyRecord: Pydantic model for organization records
- PersonRecord: Pydantic model for person records
- OrganizationDatabase: sqlite-vec database for org embedding search
- PersonDatabase: sqlite-vec database for person embedding search
- CompanyEmbedder: Embedding service using Gemma3
- Hub functions: Download/upload database from HuggingFace
"""

from .models import CompanyRecord, CompanyMatch, DatabaseStats, PersonRecord, PersonMatch, PersonType
from .store import OrganizationDatabase, get_database, PersonDatabase, get_person_database
from .embeddings import CompanyEmbedder, get_embedder
from .hub import (
    download_database,
    get_database_path,
    upload_database,
    upload_database_with_variants,
)
from .resolver import OrganizationResolver, get_organization_resolver

# Backwards compatibility alias
CompanyDatabase = OrganizationDatabase

__all__ = [
    # Organization models
    "CompanyRecord",
    "CompanyMatch",
    "DatabaseStats",
    "OrganizationDatabase",
    "CompanyDatabase",  # Backwards compatibility alias
    "get_database",
    # Person models
    "PersonRecord",
    "PersonMatch",
    "PersonType",
    "PersonDatabase",
    "get_person_database",
    # Embedding
    "CompanyEmbedder",
    "get_embedder",
    # Hub
    "download_database",
    "get_database_path",
    "upload_database",
    "upload_database_with_variants",
    # Resolver
    "OrganizationResolver",
    "get_organization_resolver",
]
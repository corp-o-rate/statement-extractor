"""
Entity/Organization database module for embedding-based entity qualification.

Provides:
- CompanyRecord: Pydantic model for organization records
- OrganizationDatabase: sqlite-vec database for embedding search
- CompanyEmbedder: Embedding service using Gemma3
- Hub functions: Download/upload database from HuggingFace
"""

from .models import CompanyRecord, CompanyMatch, DatabaseStats
from .store import OrganizationDatabase, get_database
from .embeddings import CompanyEmbedder, get_embedder
from .hub import (
    download_database,
    get_database_path,
    upload_database,
    upload_database_with_variants,
)

# Backwards compatibility alias
CompanyDatabase = OrganizationDatabase

__all__ = [
    "CompanyRecord",
    "CompanyMatch",
    "DatabaseStats",
    "OrganizationDatabase",
    "CompanyDatabase",  # Backwards compatibility alias
    "get_database",
    "CompanyEmbedder",
    "get_embedder",
    "download_database",
    "get_database_path",
    "upload_database",
    "upload_database_with_variants",
]
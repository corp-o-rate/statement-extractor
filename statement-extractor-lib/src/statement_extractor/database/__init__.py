"""
Company database module for embedding-based entity qualification.

Provides:
- CompanyRecord: Pydantic model for company records
- CompanyDatabase: sqlite-vec database for embedding search
- CompanyEmbedder: Embedding service using Gemma3
- Hub functions: Download/upload database from HuggingFace
"""

from .models import CompanyRecord, CompanyMatch, DatabaseStats
from .store import CompanyDatabase
from .embeddings import CompanyEmbedder, get_embedder
from .hub import download_database, get_database_path, upload_database

__all__ = [
    "CompanyRecord",
    "CompanyMatch",
    "DatabaseStats",
    "CompanyDatabase",
    "CompanyEmbedder",
    "get_embedder",
    "download_database",
    "get_database_path",
    "upload_database",
]
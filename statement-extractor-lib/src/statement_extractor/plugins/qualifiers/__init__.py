"""
Qualifier plugins for Stage 3 (Qualification).

Adds qualifiers and identifiers to entities.
"""

from .base import BaseQualifierPlugin
from .person import PersonQualifierPlugin

# Import embedding qualifier (may fail if database module not available)
try:
    from .embedding_company import EmbeddingCompanyQualifier
except ImportError:
    EmbeddingCompanyQualifier = None  # type: ignore

# DEPRECATED: These API-based qualifiers are deprecated in favor of EmbeddingCompanyQualifier
# They are no longer auto-registered with the plugin registry.
from .gleif import GLEIFQualifierPlugin
from .companies_house import CompaniesHouseQualifierPlugin
from .sec_edgar import SECEdgarQualifierPlugin

__all__ = [
    "BaseQualifierPlugin",
    "PersonQualifierPlugin",
    "EmbeddingCompanyQualifier",
    # Deprecated - kept for backwards compatibility
    "GLEIFQualifierPlugin",
    "CompaniesHouseQualifierPlugin",
    "SECEdgarQualifierPlugin",
]

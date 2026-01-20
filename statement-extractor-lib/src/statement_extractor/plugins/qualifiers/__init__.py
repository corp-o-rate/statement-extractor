"""
Qualifier plugins for Stage 3 (Qualification).

Adds qualifiers and identifiers to entities.
"""

from .base import BaseQualifierPlugin
from .person import PersonQualifierPlugin
from .gleif import GLEIFQualifierPlugin
from .companies_house import CompaniesHouseQualifierPlugin
from .sec_edgar import SECEdgarQualifierPlugin

# Import embedding qualifier (may fail if database module not available)
try:
    from .embedding_company import EmbeddingCompanyQualifier
except ImportError:
    EmbeddingCompanyQualifier = None  # type: ignore

__all__ = [
    "BaseQualifierPlugin",
    "PersonQualifierPlugin",
    "GLEIFQualifierPlugin",
    "CompaniesHouseQualifierPlugin",
    "SECEdgarQualifierPlugin",
    "EmbeddingCompanyQualifier",
]

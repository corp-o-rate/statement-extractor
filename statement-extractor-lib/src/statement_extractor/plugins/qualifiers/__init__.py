"""
Qualifier plugins for Stage 3 (Qualification).

Adds qualifiers and identifiers to entities.
"""

from .base import BaseQualifierPlugin
from .person import PersonQualifierPlugin
from .gleif import GLEIFQualifierPlugin
from .companies_house import CompaniesHouseQualifierPlugin
from .sec_edgar import SECEdgarQualifierPlugin

__all__ = [
    "BaseQualifierPlugin",
    "PersonQualifierPlugin",
    "GLEIFQualifierPlugin",
    "CompaniesHouseQualifierPlugin",
    "SECEdgarQualifierPlugin",
]

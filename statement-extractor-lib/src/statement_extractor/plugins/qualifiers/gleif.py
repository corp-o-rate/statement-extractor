"""
GLEIFQualifierPlugin - Qualifies ORG entities with LEI and related data.

DEPRECATED: Use EmbeddingCompanyQualifier instead, which uses a local
embedding database with pre-loaded GLEIF data for faster, offline matching.

Uses the GLEIF (Global Legal Entity Identifier Foundation) API to:
- Look up LEI by organization name
- Retrieve legal name, jurisdiction, parent company info
"""

import logging
import warnings
from typing import Optional
from urllib.parse import quote

from ..base import BaseQualifierPlugin, PluginCapability
from ...pipeline.context import PipelineContext
from ...models import ExtractedEntity, EntityQualifiers, EntityType

logger = logging.getLogger(__name__)

# GLEIF API base URL
GLEIF_API_BASE = "https://api.gleif.org/api/v1"


# DEPRECATED: Not auto-registered. Use EmbeddingCompanyQualifier instead.
class GLEIFQualifierPlugin(BaseQualifierPlugin):
    """
    DEPRECATED: Use EmbeddingCompanyQualifier instead.

    Qualifier plugin for ORG entities using GLEIF API.
    Looks up Legal Entity Identifiers (LEI) and related corporate data.
    """

    def __init__(
        self,
        timeout: int = 10,
        cache_results: bool = True,
    ):
        """
        Initialize the GLEIF qualifier.

        DEPRECATED: Use EmbeddingCompanyQualifier instead.

        Args:
            timeout: API request timeout in seconds
            cache_results: Whether to cache API results
        """
        warnings.warn(
            "GLEIFQualifierPlugin is deprecated. Use EmbeddingCompanyQualifier instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._timeout = timeout
        self._cache_results = cache_results
        self._cache: dict[str, Optional[dict]] = {}

    @property
    def name(self) -> str:
        return "gleif_qualifier"

    @property
    def priority(self) -> int:
        return 10  # High priority for ORG entities

    @property
    def capabilities(self) -> PluginCapability:
        return PluginCapability.EXTERNAL_API | PluginCapability.CACHING

    @property
    def description(self) -> str:
        return "Looks up LEI and corporate data from GLEIF API"

    @property
    def supported_entity_types(self) -> set[EntityType]:
        return {EntityType.ORG}

    @property
    def supported_identifier_types(self) -> list[str]:
        return ["lei"]  # Can lookup by existing LEI

    @property
    def provided_identifier_types(self) -> list[str]:
        return ["lei"]  # Provides LEI

    def qualify(
        self,
        entity: ExtractedEntity,
        context: PipelineContext,
    ) -> Optional[EntityQualifiers]:
        """
        Qualify an ORG entity with GLEIF data.

        Args:
            entity: The ORG entity to qualify
            context: Pipeline context

        Returns:
            EntityQualifiers with LEI and jurisdiction, or None if not found
        """
        if entity.type != EntityType.ORG:
            return None

        # Check cache first
        cache_key = entity.text.lower().strip()
        if self._cache_results and cache_key in self._cache:
            cached = self._cache[cache_key]
            if cached is None:
                return None
            return self._data_to_qualifiers(cached)

        # Search GLEIF API
        result = self._search_gleif(entity.text)

        # Cache result
        if self._cache_results:
            self._cache[cache_key] = result

        if result:
            return self._data_to_qualifiers(result)

        return None

    def _search_gleif(self, org_name: str) -> Optional[dict]:
        """Search GLEIF API for organization."""
        try:
            import requests

            # Fuzzy name search
            url = f"{GLEIF_API_BASE}/lei-records"
            params = {
                "filter[entity.legalName]": org_name,
                "page[size]": 5,
            }

            response = requests.get(url, params=params, timeout=self._timeout)
            response.raise_for_status()
            data = response.json()

            records = data.get("data", [])
            if not records:
                # Try fulltext search as fallback
                params = {
                    "filter[fulltext]": org_name,
                    "page[size]": 5,
                }
                response = requests.get(url, params=params, timeout=self._timeout)
                response.raise_for_status()
                data = response.json()
                records = data.get("data", [])

            if records:
                # Return first match
                record = records[0]
                return self._parse_lei_record(record)

        except ImportError:
            logger.warning("requests library not available for GLEIF API")
        except Exception as e:
            logger.debug(f"GLEIF API error: {e}")

        return None

    def _parse_lei_record(self, record: dict) -> dict:
        """Parse a GLEIF LEI record into a simplified dict."""
        attrs = record.get("attributes", {})
        entity = attrs.get("entity", {})
        legal_name = entity.get("legalName", {}).get("name", "")
        legal_address = entity.get("legalAddress", {})
        jurisdiction = entity.get("jurisdiction", "")

        return {
            "lei": record.get("id", ""),
            "legal_name": legal_name,
            "jurisdiction": jurisdiction,
            "country": legal_address.get("country", ""),
            "city": legal_address.get("city", ""),
            "status": attrs.get("registration", {}).get("status", ""),
        }

    def _data_to_qualifiers(self, data: dict) -> EntityQualifiers:
        """Convert GLEIF data to EntityQualifiers."""
        identifiers = {}
        if data.get("lei"):
            identifiers["lei"] = data["lei"]

        return EntityQualifiers(
            jurisdiction=data.get("jurisdiction"),
            country=data.get("country"),
            city=data.get("city"),
            identifiers=identifiers,
        )


# Allow importing without decorator for testing
GLEIFQualifierPluginClass = GLEIFQualifierPlugin

"""
CompaniesHouseQualifierPlugin - Qualifies UK ORG entities.

DEPRECATED: Use EmbeddingCompanyQualifier instead, which uses a local
embedding database with pre-loaded Companies House data for faster, offline matching.

Uses the UK Companies House API to:
- Look up company number by name
- Retrieve company details, jurisdiction, officers
"""

import logging
import os
import warnings
from typing import Optional

from ..base import BaseQualifierPlugin, PluginCapability
from ...pipeline.context import PipelineContext
from ...models import ExtractedEntity, EntityQualifiers, EntityType

logger = logging.getLogger(__name__)

# Companies House API base URL
CH_API_BASE = "https://api.company-information.service.gov.uk"


# DEPRECATED: Not auto-registered. Use EmbeddingCompanyQualifier instead.
class CompaniesHouseQualifierPlugin(BaseQualifierPlugin):
    """
    DEPRECATED: Use EmbeddingCompanyQualifier instead.

    Qualifier plugin for UK ORG entities using Companies House API.
    Requires COMPANIES_HOUSE_API_KEY environment variable.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: int = 10,
        cache_results: bool = True,
    ):
        """
        Initialize the Companies House qualifier.

        DEPRECATED: Use EmbeddingCompanyQualifier instead.

        Args:
            api_key: Companies House API key (or use COMPANIES_HOUSE_API_KEY env var)
            timeout: API request timeout in seconds
            cache_results: Whether to cache API results
        """
        warnings.warn(
            "CompaniesHouseQualifierPlugin is deprecated. Use EmbeddingCompanyQualifier instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._api_key = api_key or os.environ.get("COMPANIES_HOUSE_API_KEY")
        self._timeout = timeout
        self._cache_results = cache_results
        self._cache: dict[str, Optional[dict]] = {}

    @property
    def name(self) -> str:
        return "companies_house_qualifier"

    @property
    def priority(self) -> int:
        return 20  # Run after GLEIF

    @property
    def capabilities(self) -> PluginCapability:
        return PluginCapability.EXTERNAL_API | PluginCapability.CACHING

    @property
    def description(self) -> str:
        return "Looks up UK company data from Companies House API"

    @property
    def supported_entity_types(self) -> set[EntityType]:
        return {EntityType.ORG}

    @property
    def supported_identifier_types(self) -> list[str]:
        return ["ch_number"]  # Can lookup by company number

    @property
    def provided_identifier_types(self) -> list[str]:
        return ["ch_number"]  # Provides company number

    def qualify(
        self,
        entity: ExtractedEntity,
        context: PipelineContext,
    ) -> Optional[EntityQualifiers]:
        """
        Qualify an ORG entity with Companies House data.

        Args:
            entity: The ORG entity to qualify
            context: Pipeline context

        Returns:
            EntityQualifiers with company number, or None if not found
        """
        if entity.type != EntityType.ORG:
            return None

        if not self._api_key:
            logger.debug("Companies House API key not configured")
            return None

        # Check cache first
        cache_key = entity.text.lower().strip()
        if self._cache_results and cache_key in self._cache:
            cached = self._cache[cache_key]
            if cached is None:
                return None
            return self._data_to_qualifiers(cached)

        # Search Companies House API
        result = self._search_companies_house(entity.text)

        # Cache result
        if self._cache_results:
            self._cache[cache_key] = result

        if result:
            return self._data_to_qualifiers(result)

        return None

    def _search_companies_house(self, org_name: str) -> Optional[dict]:
        """Search Companies House API for organization."""
        try:
            import requests
            from requests.auth import HTTPBasicAuth

            url = f"{CH_API_BASE}/search/companies"
            params = {"q": org_name, "items_per_page": 5}

            response = requests.get(
                url,
                params=params,
                auth=HTTPBasicAuth(self._api_key, ""),
                timeout=self._timeout,
            )
            response.raise_for_status()
            data = response.json()

            items = data.get("items", [])
            if items:
                # Return first match
                company = items[0]
                return {
                    "ch_number": company.get("company_number", ""),
                    "title": company.get("title", ""),
                    "company_status": company.get("company_status", ""),
                    "company_type": company.get("company_type", ""),
                    "jurisdiction": "UK",
                    "country": "GB",
                    "address": company.get("address_snippet", ""),
                }

        except ImportError:
            logger.warning("requests library not available for Companies House API")
        except Exception as e:
            logger.debug(f"Companies House API error: {e}")

        return None

    def _data_to_qualifiers(self, data: dict) -> EntityQualifiers:
        """Convert Companies House data to EntityQualifiers."""
        identifiers = {}
        if data.get("ch_number"):
            identifiers["ch_number"] = data["ch_number"]

        return EntityQualifiers(
            jurisdiction=data.get("jurisdiction"),
            country=data.get("country"),
            identifiers=identifiers,
        )


# Allow importing without decorator for testing
CompaniesHouseQualifierPluginClass = CompaniesHouseQualifierPlugin

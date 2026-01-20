"""
SECEdgarQualifierPlugin - Qualifies US ORG entities with SEC data.

DEPRECATED: Use EmbeddingCompanyQualifier instead, which uses a local
embedding database with pre-loaded SEC Edgar data for faster, offline matching.

Uses the SEC EDGAR API to:
- Look up CIK (Central Index Key) by company name
- Retrieve ticker symbol, exchange, filing history
"""

import logging
import warnings
from typing import Optional

from ..base import BaseQualifierPlugin, PluginCapability
from ...pipeline.context import PipelineContext
from ...models import ExtractedEntity, EntityQualifiers, EntityType

logger = logging.getLogger(__name__)

# SEC EDGAR API endpoints
SEC_COMPANY_SEARCH = "https://efts.sec.gov/LATEST/search-index"
SEC_COMPANY_TICKERS = "https://www.sec.gov/files/company_tickers.json"


# DEPRECATED: Not auto-registered. Use EmbeddingCompanyQualifier instead.
class SECEdgarQualifierPlugin(BaseQualifierPlugin):
    """
    DEPRECATED: Use EmbeddingCompanyQualifier instead.

    Qualifier plugin for US ORG entities using SEC EDGAR.
    Provides CIK and ticker symbol for publicly traded US companies.
    """

    def __init__(
        self,
        timeout: int = 10,
        cache_results: bool = True,
    ):
        """
        Initialize the SEC EDGAR qualifier.

        DEPRECATED: Use EmbeddingCompanyQualifier instead.

        Args:
            timeout: API request timeout in seconds
            cache_results: Whether to cache API results
        """
        warnings.warn(
            "SECEdgarQualifierPlugin is deprecated. Use EmbeddingCompanyQualifier instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._timeout = timeout
        self._cache_results = cache_results
        self._cache: dict[str, Optional[dict]] = {}
        self._ticker_cache: Optional[dict] = None

    @property
    def name(self) -> str:
        return "sec_edgar_qualifier"

    @property
    def priority(self) -> int:
        return 30  # Run after GLEIF and Companies House

    @property
    def capabilities(self) -> PluginCapability:
        return PluginCapability.EXTERNAL_API | PluginCapability.CACHING

    @property
    def description(self) -> str:
        return "Looks up SEC CIK and ticker for US public companies"

    @property
    def supported_entity_types(self) -> set[EntityType]:
        return {EntityType.ORG}

    @property
    def supported_identifier_types(self) -> list[str]:
        return ["sec_cik", "ticker"]  # Can lookup by CIK or ticker

    @property
    def provided_identifier_types(self) -> list[str]:
        return ["sec_cik", "ticker"]  # Provides CIK and ticker

    def qualify(
        self,
        entity: ExtractedEntity,
        context: PipelineContext,
    ) -> Optional[EntityQualifiers]:
        """
        Qualify an ORG entity with SEC EDGAR data.

        Args:
            entity: The ORG entity to qualify
            context: Pipeline context

        Returns:
            EntityQualifiers with CIK and ticker, or None if not found
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

        # Search SEC
        result = self._search_sec(entity.text)

        # Cache result
        if self._cache_results:
            self._cache[cache_key] = result

        if result:
            return self._data_to_qualifiers(result)

        return None

    def _load_ticker_cache(self) -> dict:
        """Load the SEC company tickers JSON (cached)."""
        if self._ticker_cache is not None:
            return self._ticker_cache

        try:
            import requests

            response = requests.get(SEC_COMPANY_TICKERS, timeout=self._timeout)
            response.raise_for_status()
            data = response.json()

            # Build lookup by company name (lowercase)
            self._ticker_cache = {}
            for key, company in data.items():
                name = company.get("title", "").lower()
                if name:
                    self._ticker_cache[name] = {
                        "cik": str(company.get("cik_str", "")),
                        "ticker": company.get("ticker", ""),
                        "title": company.get("title", ""),
                    }

            logger.debug(f"Loaded {len(self._ticker_cache)} SEC company tickers")
            return self._ticker_cache

        except Exception as e:
            logger.debug(f"Failed to load SEC ticker cache: {e}")
            self._ticker_cache = {}
            return self._ticker_cache

    def _search_sec(self, org_name: str) -> Optional[dict]:
        """Search SEC for company information."""
        try:
            # Load ticker cache
            ticker_cache = self._load_ticker_cache()

            # Try exact match first
            org_lower = org_name.lower().strip()
            if org_lower in ticker_cache:
                return ticker_cache[org_lower]

            # Try partial match
            for name, data in ticker_cache.items():
                if org_lower in name or name in org_lower:
                    return data

            # Try matching without common suffixes
            clean_name = org_lower
            for suffix in [" inc", " inc.", " corp", " corp.", " co", " co.", " ltd", " llc"]:
                clean_name = clean_name.replace(suffix, "")
            clean_name = clean_name.strip()

            for name, data in ticker_cache.items():
                clean_cached = name
                for suffix in [" inc", " inc.", " corp", " corp.", " co", " co.", " ltd", " llc"]:
                    clean_cached = clean_cached.replace(suffix, "")
                clean_cached = clean_cached.strip()

                if clean_name == clean_cached or clean_name in clean_cached or clean_cached in clean_name:
                    return data

        except Exception as e:
            logger.debug(f"SEC search error: {e}")

        return None

    def _data_to_qualifiers(self, data: dict) -> EntityQualifiers:
        """Convert SEC data to EntityQualifiers."""
        identifiers = {}
        if data.get("cik"):
            identifiers["sec_cik"] = data["cik"]
        if data.get("ticker"):
            identifiers["ticker"] = data["ticker"]

        return EntityQualifiers(
            jurisdiction="US",
            country="US",
            identifiers=identifiers,
        )


# Allow importing without decorator for testing
SECEdgarQualifierPluginClass = SECEdgarQualifierPlugin

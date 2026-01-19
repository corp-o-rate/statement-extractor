"""
LocationCanonicalizer - Resolves location entities to canonical forms.

Uses:
1. ISO country code exact match
2. Known city/country mappings
3. Geohash matching for coordinates (if available)
"""

import logging
from typing import Optional

from ..base import BaseCanonicalizerPlugin, PluginCapability
from ...pipeline.context import PipelineContext
from ...pipeline.registry import PluginRegistry
from ...models import QualifiedEntity, CanonicalMatch, EntityType

logger = logging.getLogger(__name__)

# Common country name variations
COUNTRY_ALIASES = {
    "usa": "United States",
    "us": "United States",
    "united states of america": "United States",
    "u.s.": "United States",
    "u.s.a.": "United States",
    "america": "United States",
    "uk": "United Kingdom",
    "u.k.": "United Kingdom",
    "great britain": "United Kingdom",
    "britain": "United Kingdom",
    "england": "United Kingdom",
    "uae": "United Arab Emirates",
    "prc": "China",
    "peoples republic of china": "China",
    "people's republic of china": "China",
}

# ISO 3166-1 alpha-2 codes for common countries
ISO_CODES = {
    "united states": "US",
    "united kingdom": "GB",
    "china": "CN",
    "germany": "DE",
    "france": "FR",
    "japan": "JP",
    "canada": "CA",
    "australia": "AU",
    "india": "IN",
    "brazil": "BR",
    "russia": "RU",
    "italy": "IT",
    "spain": "ES",
    "mexico": "MX",
    "south korea": "KR",
    "netherlands": "NL",
    "switzerland": "CH",
    "singapore": "SG",
    "hong kong": "HK",
    "ireland": "IE",
}

# Well-known cities to countries
CITY_COUNTRY_MAP = {
    "new york": ("New York", "United States"),
    "nyc": ("New York", "United States"),
    "london": ("London", "United Kingdom"),
    "paris": ("Paris", "France"),
    "tokyo": ("Tokyo", "Japan"),
    "beijing": ("Beijing", "China"),
    "shanghai": ("Shanghai", "China"),
    "san francisco": ("San Francisco", "United States"),
    "sf": ("San Francisco", "United States"),
    "los angeles": ("Los Angeles", "United States"),
    "la": ("Los Angeles", "United States"),
    "chicago": ("Chicago", "United States"),
    "berlin": ("Berlin", "Germany"),
    "sydney": ("Sydney", "Australia"),
    "toronto": ("Toronto", "Canada"),
    "singapore": ("Singapore", "Singapore"),
    "hong kong": ("Hong Kong", "China"),
    "mumbai": ("Mumbai", "India"),
    "bangalore": ("Bangalore", "India"),
    "dublin": ("Dublin", "Ireland"),
    "amsterdam": ("Amsterdam", "Netherlands"),
    "zurich": ("Zurich", "Switzerland"),
}


def normalize_location(name: str) -> str:
    """Normalize a location name for matching."""
    return name.strip().lower().replace(".", "")


@PluginRegistry.canonicalizer
class LocationCanonicalizer(BaseCanonicalizerPlugin):
    """
    Canonicalizer for location entities (GPE, LOC).

    Uses standardized country codes and known city mappings.
    """

    @property
    def name(self) -> str:
        return "location_canonicalizer"

    @property
    def priority(self) -> int:
        return 10

    @property
    def capabilities(self) -> PluginCapability:
        return PluginCapability.CACHING

    @property
    def description(self) -> str:
        return "Resolves location entities using ISO codes and known mappings"

    @property
    def supported_entity_types(self) -> set[EntityType]:
        return {EntityType.GPE, EntityType.LOC}

    def find_canonical(
        self,
        entity: QualifiedEntity,
        context: PipelineContext,
    ) -> Optional[CanonicalMatch]:
        """
        Find canonical form for a location entity.

        Args:
            entity: Qualified entity to canonicalize
            context: Pipeline context

        Returns:
            CanonicalMatch if found
        """
        normalized = normalize_location(entity.original_text)

        # Check country aliases
        if normalized in COUNTRY_ALIASES:
            canonical_name = COUNTRY_ALIASES[normalized]
            iso_code = ISO_CODES.get(canonical_name.lower())

            return CanonicalMatch(
                canonical_id=iso_code,
                canonical_name=canonical_name,
                match_method="name_exact",
                match_confidence=1.0,
                match_details={"match_type": "country_alias"},
            )

        # Check ISO codes directly
        if normalized in ISO_CODES:
            canonical_name = normalized.title()
            iso_code = ISO_CODES[normalized]

            return CanonicalMatch(
                canonical_id=iso_code,
                canonical_name=canonical_name,
                match_method="name_exact",
                match_confidence=1.0,
                match_details={"match_type": "country_name"},
            )

        # Check city mappings
        if normalized in CITY_COUNTRY_MAP:
            city_name, country_name = CITY_COUNTRY_MAP[normalized]
            iso_code = ISO_CODES.get(country_name.lower())

            return CanonicalMatch(
                canonical_id=iso_code,
                canonical_name=city_name,
                match_method="name_exact",
                match_confidence=0.95,
                match_details={"match_type": "city_mapping", "country": country_name},
            )

        # Check qualifiers for country info
        if entity.qualifiers.country:
            country_normalized = normalize_location(entity.qualifiers.country)
            if country_normalized in ISO_CODES:
                return CanonicalMatch(
                    canonical_id=ISO_CODES[country_normalized],
                    canonical_name=entity.original_text,
                    match_method="identifier",
                    match_confidence=0.9,
                    match_details={"match_type": "qualifier_country"},
                )

        return None

    def format_fqn(
        self,
        entity: QualifiedEntity,
        match: Optional[CanonicalMatch],
    ) -> str:
        """Format FQN for a location."""
        base_name = match.canonical_name if match else entity.original_text

        parts = []

        # Add country if it's a city
        if match and match.match_details:
            country = match.match_details.get("country")
            if country:
                parts.append(country)

        # Add ISO code
        if match and match.canonical_id:
            parts.append(match.canonical_id)

        if parts:
            return f"{base_name} ({', '.join(parts)})"
        return base_name


# Allow importing without decorator for testing
LocationCanonicalizerClass = LocationCanonicalizer

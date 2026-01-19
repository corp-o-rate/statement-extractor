"""
OrganizationCanonicalizer - Resolves ORG entities to canonical forms.

Uses a tiered matching approach:
1. LEI exact match (confidence 1.0)
2. Company number + jurisdiction (confidence 0.95)
3. Trigram fuzzy name match (confidence 0.85+)
4. LLM verification for uncertain matches (confidence 0.6-0.85)
"""

import logging
import re
from typing import Optional

from ..base import BaseCanonicalizerPlugin, PluginCapability
from ...pipeline.context import PipelineContext
from ...pipeline.registry import PluginRegistry
from ...models import QualifiedEntity, CanonicalMatch, EntityType

logger = logging.getLogger(__name__)

# Common organization suffixes to normalize
ORG_SUFFIXES = [
    r'\s+Inc\.?$', r'\s+Corp\.?$', r'\s+Corporation$',
    r'\s+Ltd\.?$', r'\s+Limited$', r'\s+LLC$',
    r'\s+LLP$', r'\s+PLC$', r'\s+Co\.?$',
    r'\s+Company$', r'\s+Group$', r'\s+Holdings$',
    r'\s+&\s+Co\.?$', r',\s+Inc\.?$',
]


def normalize_org_name(name: str) -> str:
    """Normalize organization name for matching."""
    normalized = name.strip()
    for pattern in ORG_SUFFIXES:
        normalized = re.sub(pattern, '', normalized, flags=re.IGNORECASE)
    return normalized.strip().lower()


def trigram_similarity(a: str, b: str) -> float:
    """Calculate trigram similarity between two strings."""
    if not a or not b:
        return 0.0

    def get_trigrams(s: str) -> set:
        s = s.lower()
        return {s[i:i+3] for i in range(len(s) - 2)} if len(s) >= 3 else {s}

    trigrams_a = get_trigrams(a)
    trigrams_b = get_trigrams(b)

    if not trigrams_a or not trigrams_b:
        return 0.0

    intersection = len(trigrams_a & trigrams_b)
    union = len(trigrams_a | trigrams_b)

    return intersection / union if union > 0 else 0.0


@PluginRegistry.canonicalizer
class OrganizationCanonicalizer(BaseCanonicalizerPlugin):
    """
    Canonicalizer for ORG entities.

    Uses tiered matching approach with identifier, name, and fuzzy matching.
    """

    def __init__(
        self,
        fuzzy_threshold: float = 0.75,
        use_llm_verification: bool = False,
    ):
        """
        Initialize the organization canonicalizer.

        Args:
            fuzzy_threshold: Minimum trigram similarity for fuzzy matches
            use_llm_verification: Whether to use LLM for uncertain matches
        """
        self._fuzzy_threshold = fuzzy_threshold
        self._use_llm_verification = use_llm_verification

    @property
    def name(self) -> str:
        return "organization_canonicalizer"

    @property
    def priority(self) -> int:
        return 10

    @property
    def capabilities(self) -> PluginCapability:
        caps = PluginCapability.CACHING
        if self._use_llm_verification:
            caps |= PluginCapability.LLM_REQUIRED
        return caps

    @property
    def description(self) -> str:
        return "Resolves ORG entities to canonical forms using identifier and name matching"

    @property
    def supported_entity_types(self) -> set[EntityType]:
        return {EntityType.ORG}

    def find_canonical(
        self,
        entity: QualifiedEntity,
        context: PipelineContext,
    ) -> Optional[CanonicalMatch]:
        """
        Find canonical form for an ORG entity.

        Args:
            entity: Qualified entity to canonicalize
            context: Pipeline context

        Returns:
            CanonicalMatch if found
        """
        qualifiers = entity.qualifiers
        identifiers = qualifiers.identifiers

        # Tier 1: LEI exact match
        if "lei" in identifiers:
            return CanonicalMatch(
                canonical_id=identifiers["lei"],
                canonical_name=entity.original_text,
                match_method="identifier",
                match_confidence=1.0,
                match_details={"identifier_type": "lei"},
            )

        # Tier 2: Company number + jurisdiction
        if "ch_number" in identifiers and qualifiers.jurisdiction:
            return CanonicalMatch(
                canonical_id=f"{qualifiers.jurisdiction}:{identifiers['ch_number']}",
                canonical_name=entity.original_text,
                match_method="identifier",
                match_confidence=0.95,
                match_details={"identifier_type": "ch_number", "jurisdiction": qualifiers.jurisdiction},
            )

        if "sec_cik" in identifiers:
            ticker = identifiers.get("ticker", "")
            canonical_name = f"{entity.original_text} ({ticker})" if ticker else entity.original_text
            return CanonicalMatch(
                canonical_id=f"SEC:{identifiers['sec_cik']}",
                canonical_name=canonical_name,
                match_method="identifier",
                match_confidence=0.95,
                match_details={"identifier_type": "sec_cik", "ticker": ticker},
            )

        # Tier 3: Fuzzy name match against other ORG entities in context
        best_match = self._find_fuzzy_match(entity, context)
        if best_match:
            return best_match

        # No canonical match found
        return None

    def _find_fuzzy_match(
        self,
        entity: QualifiedEntity,
        context: PipelineContext,
    ) -> Optional[CanonicalMatch]:
        """Find fuzzy matches against other ORG entities in context."""
        normalized_name = normalize_org_name(entity.original_text)

        best_match = None
        best_similarity = 0.0

        for other_ref, other_entity in context.qualified_entities.items():
            if other_ref == entity.entity_ref:
                continue

            if other_entity.entity_type != EntityType.ORG:
                continue

            other_normalized = normalize_org_name(other_entity.original_text)
            similarity = trigram_similarity(normalized_name, other_normalized)

            if similarity > best_similarity and similarity >= self._fuzzy_threshold:
                best_similarity = similarity
                best_match = other_entity

        if best_match and best_similarity >= self._fuzzy_threshold:
            confidence = 0.85 + (best_similarity - self._fuzzy_threshold) * 0.1
            confidence = min(confidence, 0.95)

            return CanonicalMatch(
                canonical_id=None,
                canonical_name=best_match.original_text,
                match_method="name_fuzzy",
                match_confidence=confidence,
                match_details={"similarity": best_similarity, "matched_entity": best_match.entity_ref},
            )

        return None

    def format_fqn(
        self,
        entity: QualifiedEntity,
        match: Optional[CanonicalMatch],
    ) -> str:
        """Format FQN for an organization."""
        base_name = match.canonical_name if match else entity.original_text

        parts = []
        identifiers = entity.qualifiers.identifiers

        # Add ticker if available
        if "ticker" in identifiers:
            parts.append(identifiers["ticker"])

        # Add jurisdiction
        if entity.qualifiers.jurisdiction:
            parts.append(entity.qualifiers.jurisdiction)

        if parts:
            return f"{base_name} ({', '.join(parts)})"
        return base_name


# Allow importing without decorator for testing
OrganizationCanonicalizerClass = OrganizationCanonicalizer

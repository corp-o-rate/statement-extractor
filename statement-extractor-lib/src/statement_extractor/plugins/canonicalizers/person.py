"""
PersonCanonicalizer - Resolves PERSON entities to canonical forms.

Uses:
1. Name variants (Tim vs Timothy)
2. Role + org context matching
3. LLM identity verification for uncertain matches
"""

import logging
import re
from typing import Optional

from ..base import BaseCanonicalizerPlugin, PluginCapability
from ...pipeline.context import PipelineContext
from ...pipeline.registry import PluginRegistry
from ...models import QualifiedEntity, CanonicalMatch, EntityType

logger = logging.getLogger(__name__)

# Common name variations
NAME_VARIANTS = {
    "tim": ["timothy"],
    "timothy": ["tim"],
    "mike": ["michael"],
    "michael": ["mike"],
    "bob": ["robert"],
    "robert": ["bob", "rob"],
    "rob": ["robert"],
    "bill": ["william"],
    "william": ["bill", "will"],
    "will": ["william"],
    "jim": ["james"],
    "james": ["jim", "jimmy"],
    "jimmy": ["james"],
    "tom": ["thomas"],
    "thomas": ["tom", "tommy"],
    "joe": ["joseph"],
    "joseph": ["joe"],
    "alex": ["alexander", "alexandra"],
    "alexander": ["alex"],
    "alexandra": ["alex"],
    "dan": ["daniel"],
    "daniel": ["dan", "danny"],
    "dave": ["david"],
    "david": ["dave"],
    "ed": ["edward", "edwin"],
    "edward": ["ed", "eddie"],
    "jen": ["jennifer"],
    "jennifer": ["jen", "jenny"],
    "kate": ["katherine", "catherine"],
    "katherine": ["kate", "kathy"],
    "catherine": ["kate", "cathy"],
    "chris": ["christopher", "christine"],
    "christopher": ["chris"],
    "christine": ["chris"],
    "matt": ["matthew"],
    "matthew": ["matt"],
    "nick": ["nicholas"],
    "nicholas": ["nick"],
    "sam": ["samuel", "samantha"],
    "samuel": ["sam"],
    "samantha": ["sam"],
    "steve": ["steven", "stephen"],
    "steven": ["steve"],
    "stephen": ["steve"],
}


def normalize_person_name(name: str) -> str:
    """Normalize a person name for matching."""
    # Remove titles
    name = re.sub(r'^(Mr\.?|Mrs\.?|Ms\.?|Dr\.?|Prof\.?)\s+', '', name, flags=re.IGNORECASE)
    # Remove suffixes
    name = re.sub(r'\s+(Jr\.?|Sr\.?|III?|IV|V)$', '', name, flags=re.IGNORECASE)
    return name.strip().lower()


def get_name_parts(name: str) -> tuple[str, str]:
    """Split name into first and last name."""
    normalized = normalize_person_name(name)
    parts = normalized.split()
    if len(parts) >= 2:
        return parts[0], parts[-1]
    elif len(parts) == 1:
        return parts[0], ""
    return "", ""


def names_match(name1: str, name2: str) -> tuple[bool, float, bool]:
    """
    Check if two names match, considering variants.

    Returns (matches, confidence, is_variant).
    """
    first1, last1 = get_name_parts(name1)
    first2, last2 = get_name_parts(name2)

    # Last names must match (if both present)
    if last1 and last2 and last1 != last2:
        return False, 0.0, False

    # Check first name match
    if first1 == first2:
        return True, 1.0, False

    # Check variants
    variants1 = NAME_VARIANTS.get(first1, [])
    variants2 = NAME_VARIANTS.get(first2, [])

    if first2 in variants1 or first1 in variants2:
        return True, 0.9, True

    return False, 0.0, False


@PluginRegistry.canonicalizer
class PersonCanonicalizer(BaseCanonicalizerPlugin):
    """
    Canonicalizer for PERSON entities.

    Uses name variants and context matching.
    """

    def __init__(
        self,
        use_context_matching: bool = True,
    ):
        """
        Initialize the person canonicalizer.

        Args:
            use_context_matching: Whether to use role+org for disambiguation
        """
        self._use_context_matching = use_context_matching

    @property
    def name(self) -> str:
        return "person_canonicalizer"

    @property
    def priority(self) -> int:
        return 10

    @property
    def capabilities(self) -> PluginCapability:
        return PluginCapability.CACHING

    @property
    def description(self) -> str:
        return "Resolves PERSON entities using name variants and context"

    @property
    def supported_entity_types(self) -> set[EntityType]:
        return {EntityType.PERSON}

    def find_canonical(
        self,
        entity: QualifiedEntity,
        context: PipelineContext,
    ) -> Optional[CanonicalMatch]:
        """
        Find canonical form for a PERSON entity.

        Args:
            entity: Qualified entity to canonicalize
            context: Pipeline context

        Returns:
            CanonicalMatch if found
        """
        # Look for matching PERSON entities in context
        best_match = None
        best_confidence = 0.0

        best_is_variant = False

        for other_ref, other_entity in context.qualified_entities.items():
            if other_ref == entity.entity_ref:
                continue

            if other_entity.entity_type != EntityType.PERSON:
                continue

            # Check name match
            matches, confidence, is_variant = names_match(entity.original_text, other_entity.original_text)
            if not matches:
                continue

            # Boost confidence if role+org also match
            if self._use_context_matching and confidence > 0:
                my_qualifiers = entity.qualifiers
                other_qualifiers = other_entity.qualifiers

                if my_qualifiers.role and other_qualifiers.role:
                    if my_qualifiers.role.lower() == other_qualifiers.role.lower():
                        confidence = min(confidence + 0.05, 1.0)

                if my_qualifiers.org and other_qualifiers.org:
                    if my_qualifiers.org.lower() == other_qualifiers.org.lower():
                        confidence = min(confidence + 0.05, 1.0)

            if confidence > best_confidence:
                best_confidence = confidence
                best_match = other_entity
                best_is_variant = is_variant

        if best_match and best_confidence >= 0.8:
            return CanonicalMatch(
                canonical_id=None,
                canonical_name=best_match.original_text,
                match_method="name_variant" if best_is_variant else "name_exact",
                match_confidence=best_confidence,
                match_details={"matched_entity": best_match.entity_ref},
            )

        return None

    def format_fqn(
        self,
        entity: QualifiedEntity,
        match: Optional[CanonicalMatch],
    ) -> str:
        """Format FQN for a person."""
        base_name = match.canonical_name if match else entity.original_text

        parts = []
        qualifiers = entity.qualifiers

        if qualifiers.role:
            parts.append(qualifiers.role)

        if qualifiers.org:
            parts.append(qualifiers.org)

        if parts:
            return f"{base_name} ({', '.join(parts)})"
        return base_name


# Allow importing without decorator for testing
PersonCanonicalizerClass = PersonCanonicalizer

"""
PersonQualifierPlugin - Qualifies PERSON entities with role and organization.

Uses Gemma3 12B (instruction-tuned) to extract:
- role: Job title/position (e.g., "CEO", "President")
- org: Organization/employer (e.g., "Apple Inc", "Microsoft")
"""

import json
import logging
import re
from typing import Optional

from ..base import BaseQualifierPlugin, PluginCapability
from ...pipeline.context import PipelineContext
from ...pipeline.registry import PluginRegistry
from ...models import (
    ExtractedEntity,
    EntityQualifiers,
    EntityType,
    QualifiedEntity,
    CanonicalEntity,
)
from ...llm import LLM

logger = logging.getLogger(__name__)


@PluginRegistry.qualifier
class PersonQualifierPlugin(BaseQualifierPlugin):
    """
    Qualifier plugin for PERSON entities.

    Uses Gemma3 12B to extract role and organization from context.
    Falls back to pattern matching if model is not available.
    """

    # Common role patterns for fallback
    ROLE_PATTERNS = [
        r"\b(CEO|CFO|CTO|COO|CMO|CIO|CISO|CSO)\b",
        r"\b(Chief\s+\w+\s+Officer)\b",
        r"\b(President|Chairman|Director|Manager|Executive|Founder|Co-Founder)\b",
        r"\b(Vice\s+President|VP)\b",
        r"\b(Head\s+of\s+\w+)\b",
        r"\b(Senior\s+\w+|Lead\s+\w+|Principal\s+\w+)\b",
    ]

    def __init__(
        self,
        model_id: str = "google/gemma-3-12b-it-qat-q4_0-gguf",
        gguf_file: Optional[str] = None,
        use_llm: bool = True,
        use_4bit: bool = True,
    ):
        """
        Initialize the person qualifier.

        Args:
            model_id: HuggingFace model ID for LLM qualification
            gguf_file: GGUF filename for quantized models (auto-detected if model_id ends with -gguf)
            use_llm: Whether to use LLM
            use_4bit: Use 4-bit quantization (requires bitsandbytes, ignored for GGUF)
        """
        self._use_llm = use_llm
        self._llm: Optional[LLM] = None
        if use_llm:
            self._llm = LLM(
                model_id=model_id,
                gguf_file=gguf_file,
                use_4bit=use_4bit,
            )

    @property
    def name(self) -> str:
        return "person_qualifier"

    @property
    def priority(self) -> int:
        return 10  # High priority for PERSON entities

    @property
    def capabilities(self) -> PluginCapability:
        caps = PluginCapability.NONE
        if self._use_llm:
            caps |= PluginCapability.LLM_REQUIRED
        return caps

    @property
    def description(self) -> str:
        return "Extracts role and organization for PERSON entities using Gemma3"

    @property
    def supported_entity_types(self) -> set[EntityType]:
        return {EntityType.PERSON}

    @property
    def provided_identifier_types(self) -> list[str]:
        return []  # Provides qualifiers, not identifiers

    def qualify(
        self,
        entity: ExtractedEntity,
        context: PipelineContext,
    ) -> Optional[CanonicalEntity]:
        """
        Qualify a PERSON entity with role and organization.

        Args:
            entity: The PERSON entity to qualify
            context: Pipeline context for accessing source text

        Returns:
            CanonicalEntity with role/org qualifiers and FQN, or None if nothing found
        """
        if entity.type != EntityType.PERSON:
            return None

        # Use the full source text for LLM qualification
        # This provides maximum context for understanding the person's role/org
        full_text = context.source_text

        # Try LLM extraction first with full text
        qualifiers: Optional[EntityQualifiers] = None
        if self._llm is not None:
            result = self._extract_with_llm(entity.text, full_text)
            if result and (result.role or result.org):
                qualifiers = result

        # Fallback to pattern matching with full text
        if qualifiers is None:
            qualifiers = self._extract_with_patterns(entity.text, full_text)

        if qualifiers is None:
            return None

        # Build CanonicalEntity from qualifiers
        return self._build_canonical_entity(entity, qualifiers)

    def _extract_with_llm(
        self,
        person_name: str,
        context_text: str,
    ) -> Optional[EntityQualifiers]:
        """Extract role and org using Gemma3."""
        if self._llm is None:
            return None

        try:
            prompt = f"""Extract qualifiers for a person from the given context.
Instructions:
- "role" = job title or position (e.g., "CEO", "President", "Director")
- "org" = company or organization name (e.g., "Amazon", "Apple Inc", "Microsoft")
- These are DIFFERENT things: role is a job title, org is a company name
- Return null for fields not mentioned in the context

Return ONLY valid JSON:

E.g.
<context>We interviewed Big Ducks Quacking Inc team. James is new in the role of the CEO</context>
<person>James</person>

Should return:

{{"role": "CEO", "org": "Big Ducks Quacking Inc"}}

---

<context>{context_text}</context>
<person>{person_name}</person>
"""

            logger.debug(f"LLM request: {prompt}")
            response = self._llm.generate(prompt, max_tokens=100, stop=["\n\n", "</s>"])
            logger.debug(f"LLM response: {response}")

            # Extract JSON from response
            json_match = re.search(r'\{[^}]+\}', response)
            if json_match:
                data = json.loads(json_match.group())
                role = data.get("role")
                org = data.get("org")

                # Validate: role and org should be different (reject if same)
                if role and org and role.lower() == org.lower():
                    logger.debug(f"Rejected duplicate role/org: {role}")
                    org = None  # Clear org if it's same as role

                if role or org:
                    return EntityQualifiers(role=role, org=org)

        except Exception as e:
            logger.exception(f"LLM extraction failed: {e}")
            raise e

        return None

    def _extract_with_patterns(
        self,
        person_name: str,
        context_text: str,
    ) -> Optional[EntityQualifiers]:
        """Extract role and org using pattern matching."""
        role = None
        org = None

        # Look for role patterns
        for pattern in self.ROLE_PATTERNS:
            match = re.search(pattern, context_text, re.IGNORECASE)
            if match:
                role = match.group(1)
                break

        # Look for "of [Organization]" or "at [Organization]" patterns
        org_patterns = [
            rf'{re.escape(person_name)}[^.]*?\bof\s+([A-Z][A-Za-z\s&]+(?:Inc|Corp|Ltd|LLC|Company|Co)?\.?)',
            rf'{re.escape(person_name)}[^.]*?\bat\s+([A-Z][A-Za-z\s&]+(?:Inc|Corp|Ltd|LLC|Company|Co)?\.?)',
            rf'([A-Z][A-Za-z\s&]+(?:Inc|Corp|Ltd|LLC|Company|Co)?\.?)\s*(?:\'s|s)?\s*{re.escape(person_name)}',
        ]

        for pattern in org_patterns:
            match = re.search(pattern, context_text)
            if match:
                org = match.group(1).strip()
                # Clean up trailing punctuation
                org = org.rstrip('.,;')
                break

        if role or org:
            return EntityQualifiers(role=role, org=org)

        return None

    def _build_canonical_entity(
        self,
        entity: ExtractedEntity,
        qualifiers: EntityQualifiers,
    ) -> CanonicalEntity:
        """Build CanonicalEntity from qualifiers."""
        # Create QualifiedEntity
        qualified = QualifiedEntity(
            entity_ref=entity.entity_ref,
            original_text=entity.text,
            entity_type=entity.type,
            qualifiers=qualifiers,
            qualification_sources=[self.name],
        )

        # Build FQN: "Person Name (Role, Org)" or "Person Name (Role)" or "Person Name (Org)"
        fqn_parts = []
        if qualifiers.role:
            fqn_parts.append(qualifiers.role)
        if qualifiers.org:
            fqn_parts.append(qualifiers.org)

        if fqn_parts:
            fqn = f"{entity.text} ({', '.join(fqn_parts)})"
        else:
            fqn = entity.text

        return CanonicalEntity(
            entity_ref=entity.entity_ref,
            qualified_entity=qualified,
            canonical_match=None,  # PERSON entities don't have canonical matches
            fqn=fqn,
        )


# Allow importing without decorator for testing
PersonQualifierPluginClass = PersonQualifierPlugin

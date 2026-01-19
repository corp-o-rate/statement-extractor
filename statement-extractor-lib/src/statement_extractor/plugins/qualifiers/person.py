"""
PersonQualifierPlugin - Qualifies PERSON entities with role and organization.

Uses Gemma3 1B (instruction-tuned) to extract:
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
from ...models import ExtractedEntity, EntityQualifiers, EntityType

logger = logging.getLogger(__name__)


@PluginRegistry.qualifier
class PersonQualifierPlugin(BaseQualifierPlugin):
    """
    Qualifier plugin for PERSON entities.

    Uses Gemma3 1B to extract role and organization from context.
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
        model_id: str = "google/gemma-3-1b-it",
        use_llm: bool = True,
        use_4bit: bool = True,
    ):
        """
        Initialize the person qualifier.

        Args:
            model_id: HuggingFace model ID for LLM qualification
            use_llm: Whether to use LLM (False = pattern matching only)
            use_4bit: Use 4-bit quantization (requires bitsandbytes)
        """
        self._model_id = model_id
        self._use_llm = use_llm
        self._use_4bit = use_4bit
        self._model = None
        self._tokenizer = None

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

    def _load_model(self):
        """Lazy-load the Gemma3 model."""
        if self._model is not None:
            return

        if not self._use_llm:
            return

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            logger.info(f"Loading {self._model_id} for person qualification...")

            self._tokenizer = AutoTokenizer.from_pretrained(self._model_id)

            if self._use_4bit:
                try:
                    from transformers import BitsAndBytesConfig
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                    )
                    self._model = AutoModelForCausalLM.from_pretrained(
                        self._model_id,
                        quantization_config=quantization_config,
                        device_map="auto",
                    )
                except ImportError:
                    logger.warning("bitsandbytes not available, loading full precision")
                    self._model = AutoModelForCausalLM.from_pretrained(
                        self._model_id,
                        device_map="auto",
                        torch_dtype=torch.float16,
                    )
            else:
                self._model = AutoModelForCausalLM.from_pretrained(
                    self._model_id,
                    device_map="auto",
                    torch_dtype=torch.float16,
                )

            logger.debug("Gemma3 model loaded")

        except Exception as e:
            logger.warning(f"Failed to load Gemma3 model: {e}")
            self._use_llm = False

    def qualify(
        self,
        entity: ExtractedEntity,
        context: PipelineContext,
    ) -> Optional[EntityQualifiers]:
        """
        Qualify a PERSON entity with role and organization.

        Args:
            entity: The PERSON entity to qualify
            context: Pipeline context for accessing source text

        Returns:
            EntityQualifiers with role and org, or None if nothing found
        """
        if entity.type != EntityType.PERSON:
            return None

        # Find statements involving this entity
        relevant_statements = [
            stmt for stmt in context.statements
            if stmt.subject.entity_ref == entity.entity_ref or
               stmt.object.entity_ref == entity.entity_ref
        ]

        if not relevant_statements:
            return None

        # Combine source texts for context
        source_texts = [stmt.source_text for stmt in relevant_statements]
        combined_context = " ".join(source_texts)

        # Try LLM extraction first
        if self._use_llm:
            self._load_model()
            if self._model is not None:
                result = self._extract_with_llm(entity.text, combined_context)
                if result and (result.role or result.org):
                    return result

        # Fallback to pattern matching
        return self._extract_with_patterns(entity.text, combined_context)

    def _extract_with_llm(
        self,
        person_name: str,
        context_text: str,
    ) -> Optional[EntityQualifiers]:
        """Extract role and org using Gemma3."""
        try:
            prompt = f"""Given the context, extract qualifiers for the person.
Context: {context_text}
Person: {person_name}

Extract the person's job title/role and their organization/employer from the context.
Return ONLY a JSON object with these fields:
- "role": the job title or role (null if not found)
- "org": the organization or employer (null if not found)

JSON:"""

            inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)

            with self._model.no_grad() if hasattr(self._model, 'no_grad') else __import__('contextlib').nullcontext():
                import torch
                with torch.no_grad():
                    outputs = self._model.generate(
                        **inputs,
                        max_new_tokens=100,
                        do_sample=False,
                        pad_token_id=self._tokenizer.pad_token_id,
                    )

            response = self._tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract JSON from response
            json_match = re.search(r'\{[^}]+\}', response)
            if json_match:
                data = json.loads(json_match.group())
                role = data.get("role")
                org = data.get("org")

                if role or org:
                    return EntityQualifiers(role=role, org=org)

        except Exception as e:
            logger.debug(f"LLM extraction failed: {e}")

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


# Allow importing without decorator for testing
PersonQualifierPluginClass = PersonQualifierPlugin

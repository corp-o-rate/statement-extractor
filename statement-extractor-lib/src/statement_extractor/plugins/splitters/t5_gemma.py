"""
T5GemmaSplitter - Stage 1 plugin that wraps the existing StatementExtractor.

Uses T5-Gemma2 model with Diverse Beam Search to generate high-quality
subject-predicate-object triples from text.
"""

import logging
import re
import xml.etree.ElementTree as ET
from typing import Optional

from ..base import BaseSplitterPlugin, PluginCapability
from ...pipeline.context import PipelineContext
from ...pipeline.registry import PluginRegistry
from ...models import RawTriple

logger = logging.getLogger(__name__)


@PluginRegistry.splitter
class T5GemmaSplitter(BaseSplitterPlugin):
    """
    Splitter plugin that uses T5-Gemma2 for triple extraction.

    Wraps the existing StatementExtractor from extractor.py to produce
    RawTriple objects for the pipeline.
    """

    def __init__(
        self,
        model_id: Optional[str] = None,
        device: Optional[str] = None,
        num_beams: int = 4,
        diversity_penalty: float = 1.0,
        max_new_tokens: int = 2048,
    ):
        """
        Initialize the T5Gemma splitter.

        Args:
            model_id: HuggingFace model ID (defaults to Corp-o-Rate model)
            device: Device to use (auto-detected if not specified)
            num_beams: Number of beams for diverse beam search
            diversity_penalty: Penalty for beam diversity
            max_new_tokens: Maximum tokens to generate
        """
        self._model_id = model_id
        self._device = device
        self._num_beams = num_beams
        self._diversity_penalty = diversity_penalty
        self._max_new_tokens = max_new_tokens
        self._extractor = None

    @property
    def name(self) -> str:
        return "t5_gemma_splitter"

    @property
    def priority(self) -> int:
        return 10  # High priority - primary splitter

    @property
    def capabilities(self) -> PluginCapability:
        return PluginCapability.LLM_REQUIRED

    @property
    def description(self) -> str:
        return "T5-Gemma2 model for extracting triples using Diverse Beam Search"

    def _get_extractor(self):
        """Lazy-load the StatementExtractor."""
        if self._extractor is None:
            from ...extractor import StatementExtractor
            # Only pass model_id and device if they were explicitly set
            kwargs = {}
            if self._model_id is not None:
                kwargs["model_id"] = self._model_id
            if self._device is not None:
                kwargs["device"] = self._device
            self._extractor = StatementExtractor(**kwargs)
        return self._extractor

    def split(
        self,
        text: str,
        context: PipelineContext,
    ) -> list[RawTriple]:
        """
        Split text into raw triples using T5-Gemma2.

        Args:
            text: Input text to split
            context: Pipeline context

        Returns:
            List of RawTriple objects
        """
        logger.debug(f"T5GemmaSplitter processing {len(text)} chars")

        # Get options from context if available
        splitter_options = context.source_metadata.get("splitter_options", {})
        num_beams = splitter_options.get("num_beams", self._num_beams)
        diversity_penalty = splitter_options.get("diversity_penalty", self._diversity_penalty)
        max_new_tokens = splitter_options.get("max_new_tokens", self._max_new_tokens)

        # Create extraction options
        from ...models import ExtractionOptions as LegacyExtractionOptions
        options = LegacyExtractionOptions(
            num_beams=num_beams,
            diversity_penalty=diversity_penalty,
            max_new_tokens=max_new_tokens,
            # Disable GLiNER and dedup - we handle those in later stages
            use_gliner_extraction=False,
            embedding_dedup=False,
            deduplicate=False,
        )

        # Get raw XML from extractor
        extractor = self._get_extractor()
        xml_output = extractor.extract_as_xml(text, options)

        # Parse XML to RawTriple objects
        raw_triples = self._parse_xml_to_raw_triples(xml_output)

        logger.info(f"T5GemmaSplitter produced {len(raw_triples)} raw triples")
        return raw_triples

    def _parse_xml_to_raw_triples(self, xml_output: str) -> list[RawTriple]:
        """Parse XML output into RawTriple objects."""
        raw_triples = []

        try:
            root = ET.fromstring(xml_output)
        except ET.ParseError as e:
            logger.warning(f"XML parse error: {e}")
            # Try to repair
            xml_output = self._repair_xml(xml_output)
            try:
                root = ET.fromstring(xml_output)
            except ET.ParseError:
                logger.error("XML repair failed")
                return raw_triples

        if root.tag != "statements":
            logger.warning(f"Unexpected root tag: {root.tag}")
            return raw_triples

        for stmt_elem in root.findall("stmt"):
            try:
                subject_elem = stmt_elem.find("subject")
                predicate_elem = stmt_elem.find("predicate")
                object_elem = stmt_elem.find("object")
                text_elem = stmt_elem.find("text")

                subject_text = subject_elem.text.strip() if subject_elem is not None and subject_elem.text else ""
                predicate_text = predicate_elem.text.strip() if predicate_elem is not None and predicate_elem.text else ""
                object_text = object_elem.text.strip() if object_elem is not None and object_elem.text else ""
                source_text = text_elem.text.strip() if text_elem is not None and text_elem.text else ""

                if subject_text and object_text and source_text:
                    raw_triples.append(RawTriple(
                        subject_text=subject_text,
                        predicate_text=predicate_text,
                        object_text=object_text,
                        source_sentence=source_text,
                    ))
                else:
                    logger.debug(f"Skipping incomplete triple: s={subject_text}, p={predicate_text}, o={object_text}")

            except Exception as e:
                logger.warning(f"Error parsing stmt element: {e}")
                continue

        return raw_triples

    def _repair_xml(self, xml_string: str) -> str:
        """Attempt to repair common XML syntax errors."""
        # Use the repair function from extractor.py
        from ...extractor import repair_xml
        repaired, repairs = repair_xml(xml_string)
        if repairs:
            logger.debug(f"XML repairs: {', '.join(repairs)}")
        return repaired


# Allow importing without decorator for testing
T5GemmaSplitterClass = T5GemmaSplitter

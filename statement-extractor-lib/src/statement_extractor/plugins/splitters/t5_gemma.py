"""
T5GemmaSplitter - Stage 1 plugin that splits text into atomic sentences.

Uses T5-Gemma2 model with Diverse Beam Search to split unstructured text
into atomic statements that can be converted to triples in Stage 2.
"""

import logging
import re
from typing import Optional

from ..base import BaseSplitterPlugin, PluginCapability
from ...pipeline.context import PipelineContext
from ...pipeline.registry import PluginRegistry
from ...models import SplitSentence

logger = logging.getLogger(__name__)


@PluginRegistry.splitter
class T5GemmaSplitter(BaseSplitterPlugin):
    """
    Splitter plugin that uses T5-Gemma2 to split text into atomic sentences.

    Uses the T5-Gemma2 model to identify and extract atomic statements
    from unstructured text. Each sentence can be converted to a
    subject-predicate-object triple in Stage 2.
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
        return PluginCapability.LLM_REQUIRED | PluginCapability.BATCH_PROCESSING

    @property
    def description(self) -> str:
        return "T5-Gemma2 model for splitting text into atomic sentences"

    @property
    def model_vram_gb(self) -> float:
        """T5-Gemma2 model weights ~2GB in bfloat16."""
        return 2.0

    @property
    def per_item_vram_gb(self) -> float:
        """Each text item during batch processing ~0.5GB for KV cache and activations."""
        return 0.5

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
    ) -> list[SplitSentence]:
        """
        Split text into atomic sentences using T5-Gemma2.

        Args:
            text: Input text to split
            context: Pipeline context

        Returns:
            List of SplitSentence objects
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

        # Parse XML to SplitSentence objects
        sentences = self._parse_xml_to_sentences(xml_output)

        logger.info(f"T5GemmaSplitter produced {len(sentences)} sentences")
        return sentences

    def split_batch(
        self,
        texts: list[str],
        context: PipelineContext,
    ) -> list[list[SplitSentence]]:
        """
        Split multiple texts into atomic sentences using batch processing.

        Processes all texts through the T5-Gemma2 model in batches
        sized for optimal GPU utilization.

        Args:
            texts: List of input texts to split
            context: Pipeline context

        Returns:
            List of SplitSentence lists, one per input text
        """
        if not texts:
            return []

        batch_size = self.get_optimal_batch_size()
        logger.info(f"T5GemmaSplitter batch processing {len(texts)} texts with batch_size={batch_size}")

        # Get options from context
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
            use_gliner_extraction=False,
            embedding_dedup=False,
            deduplicate=False,
        )

        extractor = self._get_extractor()
        all_results: list[list[SplitSentence]] = []

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            logger.debug(f"Processing batch {i // batch_size + 1}: {len(batch_texts)} texts")

            batch_results = self._process_batch(batch_texts, extractor, options)
            all_results.extend(batch_results)

        total_sentences = sum(len(r) for r in all_results)
        logger.info(f"T5GemmaSplitter batch produced {total_sentences} total sentences from {len(texts)} texts")
        return all_results

    def _process_batch(
        self,
        texts: list[str],
        extractor,
        options,
    ) -> list[list[SplitSentence]]:
        """
        Process a batch of texts through the model.

        Uses the model's batch generation capability for efficient GPU utilization.
        """
        import torch

        # Wrap texts in page tags
        wrapped_texts = [f"<page>{t}</page>" if not t.startswith("<page>") else t for t in texts]

        # Tokenize batch
        tokenizer = extractor.tokenizer
        model = extractor.model

        inputs = tokenizer(
            wrapped_texts,
            return_tensors="pt",
            max_length=4096,
            truncation=True,
            padding=True,
        ).to(extractor.device)

        # Create stopping criteria
        from ...extractor import StopOnSequence
        from transformers import StoppingCriteriaList

        input_length = inputs["input_ids"].shape[1]
        stop_criteria = StopOnSequence(
            tokenizer=tokenizer,
            stop_sequence="</statements>",
            input_length=input_length,
        )

        # Generate for all texts in batch
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=options.max_new_tokens,
                max_length=None,
                num_beams=options.num_beams,
                num_beam_groups=options.num_beams,
                num_return_sequences=1,  # One sequence per input for batch
                diversity_penalty=options.diversity_penalty,
                do_sample=False,
                top_p=None,
                top_k=None,
                trust_remote_code=True,
                custom_generate="transformers-community/group-beam-search",
                stopping_criteria=StoppingCriteriaList([stop_criteria]),
            )

        # Decode and parse each output
        results: list[list[SplitSentence]] = []
        end_tag = "</statements>"

        for output in outputs:
            decoded = tokenizer.decode(output, skip_special_tokens=True)

            # Truncate at </statements>
            if end_tag in decoded:
                end_pos = decoded.find(end_tag) + len(end_tag)
                decoded = decoded[:end_pos]

            sentences = self._parse_xml_to_sentences(decoded)
            results.append(sentences)

        return results

    # Regex pattern to extract <text> content from <stmt> blocks
    _STMT_TEXT_PATTERN = re.compile(r'<stmt>.*?<text>(.*?)</text>.*?</stmt>', re.DOTALL)

    def _parse_xml_to_sentences(self, xml_output: str) -> list[SplitSentence]:
        """Extract atomic sentences from <stmt><text>...</text></stmt> blocks."""
        sentences = []

        # Find all <text> content within <stmt> blocks
        text_matches = self._STMT_TEXT_PATTERN.findall(xml_output)
        logger.debug(f"Found {len(text_matches)} stmt text blocks via regex")

        for sentence_text in text_matches:
            sentence_text = sentence_text.strip()
            if sentence_text:
                sentences.append(SplitSentence(text=sentence_text))

        return sentences


# Allow importing without decorator for testing
T5GemmaSplitterClass = T5GemmaSplitter

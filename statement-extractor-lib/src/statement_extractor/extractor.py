"""
Statement Extractor - Extract structured statements from text using T5-Gemma 2.

This module uses Diverse Beam Search (Vijayakumar et al., 2016) to generate
multiple candidate extractions and selects/merges the best results using
quality scoring.

Paper: https://arxiv.org/abs/1610.02424
"""

import logging
import re
import xml.etree.ElementTree as ET
from typing import Optional

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from .models import (
    Entity,
    EntityType,
    ExtractionOptions,
    ExtractionResult,
    PredicateComparisonConfig,
    PredicateTaxonomy,
    ScoringConfig,
    Statement,
)

logger = logging.getLogger(__name__)

# Default model
DEFAULT_MODEL_ID = "Corp-o-Rate-Community/statement-extractor"


class StatementExtractor:
    """
    Extract structured statements from unstructured text.

    Uses the T5-Gemma 2 statement extraction model with Diverse Beam Search
    to generate high-quality subject-predicate-object triples.

    Features:
    - Quality-based beam scoring (not just longest output)
    - Beam merging for better coverage
    - Embedding-based predicate comparison for smart deduplication
    - Configurable precision/recall tradeoff

    Example:
        >>> extractor = StatementExtractor()
        >>> result = extractor.extract("Apple Inc. announced a new iPhone today.")
        >>> for stmt in result:
        ...     print(stmt)
        Apple Inc. -- announced --> a new iPhone
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        predicate_taxonomy: Optional[PredicateTaxonomy] = None,
        predicate_config: Optional[PredicateComparisonConfig] = None,
        scoring_config: Optional[ScoringConfig] = None,
    ):
        """
        Initialize the statement extractor.

        Args:
            model_id: HuggingFace model ID or local path
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
            torch_dtype: Torch dtype (default: bfloat16 on GPU, float32 on CPU)
            predicate_taxonomy: Optional taxonomy for predicate normalization
            predicate_config: Configuration for predicate comparison
            scoring_config: Configuration for quality scoring
        """
        self.model_id = model_id
        self._model: Optional[AutoModelForSeq2SeqLM] = None
        self._tokenizer: Optional[AutoTokenizer] = None

        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Auto-detect dtype
        if torch_dtype is None:
            self.torch_dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        else:
            self.torch_dtype = torch_dtype

        # Scoring and comparison config
        self._predicate_taxonomy = predicate_taxonomy
        self._predicate_config = predicate_config
        self._scoring_config = scoring_config

        # Lazy-loaded components
        self._beam_scorer = None
        self._predicate_comparer = None

    def _load_model(self) -> None:
        """Load model and tokenizer if not already loaded."""
        if self._model is not None:
            return

        logger.info(f"Loading model: {self.model_id}")

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=True,
        )

        if self.device == "cuda":
            self._model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_id,
                torch_dtype=self.torch_dtype,
                trust_remote_code=True,
                device_map="auto",
            )
        else:
            self._model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_id,
                trust_remote_code=True,
            )
            self._model = self._model.to(self.device)

        logger.info(f"Model loaded on {self.device}")

    def _get_beam_scorer(self, options: ExtractionOptions):
        """Get or create beam scorer with current config."""
        from .scoring import BeamScorer

        config = options.scoring_config or self._scoring_config or ScoringConfig()
        return BeamScorer(config=config)

    def _get_predicate_comparer(self, options: ExtractionOptions):
        """Get or create predicate comparer if embeddings enabled."""
        if not options.embedding_dedup:
            return None

        from .predicate_comparer import PredicateComparer

        taxonomy = options.predicate_taxonomy or self._predicate_taxonomy
        config = options.predicate_config or self._predicate_config or PredicateComparisonConfig()
        return PredicateComparer(taxonomy=taxonomy, config=config)

    @property
    def model(self) -> AutoModelForSeq2SeqLM:
        """Get the model, loading it if necessary."""
        self._load_model()
        return self._model

    @property
    def tokenizer(self) -> AutoTokenizer:
        """Get the tokenizer, loading it if necessary."""
        self._load_model()
        return self._tokenizer

    def extract(
        self,
        text: str,
        options: Optional[ExtractionOptions] = None,
    ) -> ExtractionResult:
        """
        Extract statements from text.

        Args:
            text: Input text to extract statements from
            options: Extraction options (uses defaults if not provided)

        Returns:
            ExtractionResult containing the extracted statements
        """
        if options is None:
            options = ExtractionOptions()

        # Store original text for scoring
        original_text = text

        # Wrap text in page tags if not already wrapped
        if not text.startswith("<page>"):
            text = f"<page>{text}</page>"

        # Run extraction with retry logic
        statements = self._extract_with_scoring(text, original_text, options)

        return ExtractionResult(
            statements=statements,
            source_text=original_text,
        )

    def extract_as_xml(
        self,
        text: str,
        options: Optional[ExtractionOptions] = None,
    ) -> str:
        """
        Extract statements and return raw XML output.

        Note: This bypasses the new scoring/merging logic for backward compatibility.
        Use extract() for full quality scoring.

        Args:
            text: Input text to extract statements from
            options: Extraction options

        Returns:
            XML string with <statements> containing <stmt> elements
        """
        if options is None:
            options = ExtractionOptions()

        if not text.startswith("<page>"):
            text = f"<page>{text}</page>"

        return self._extract_raw_xml(text, options)

    def extract_as_json(
        self,
        text: str,
        options: Optional[ExtractionOptions] = None,
        indent: Optional[int] = 2,
    ) -> str:
        """
        Extract statements and return JSON string.

        Args:
            text: Input text to extract statements from
            options: Extraction options
            indent: JSON indentation (None for compact)

        Returns:
            JSON string representation of the extraction result
        """
        result = self.extract(text, options)
        return result.model_dump_json(indent=indent)

    def extract_as_dict(
        self,
        text: str,
        options: Optional[ExtractionOptions] = None,
    ) -> dict:
        """
        Extract statements and return as dictionary.

        Args:
            text: Input text to extract statements from
            options: Extraction options

        Returns:
            Dictionary representation of the extraction result
        """
        result = self.extract(text, options)
        return result.model_dump()

    def _extract_with_scoring(
        self,
        text: str,
        original_text: str,
        options: ExtractionOptions,
    ) -> list[Statement]:
        """
        Extract statements with quality scoring and beam merging.

        This is the new extraction pipeline that:
        1. Generates multiple candidates via DBS
        2. Parses each to statements
        3. Scores each triple for groundedness
        4. Merges top beams or selects best beam
        5. Deduplicates using embeddings (if enabled)
        """
        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=4096,
            truncation=True,
        ).to(self.device)

        # Count sentences for quality check
        num_sentences = self._count_sentences(text)
        min_expected = int(num_sentences * options.min_statement_ratio)

        logger.info(f"Input has ~{num_sentences} sentences, expecting >= {min_expected} statements")

        # Get beam scorer
        beam_scorer = self._get_beam_scorer(options)

        all_candidates: list[list[Statement]] = []

        for attempt in range(options.max_attempts):
            # Generate candidate beams
            candidates = self._generate_candidate_beams(inputs, options)

            # Parse each candidate to statements
            parsed_candidates = []
            for xml_output in candidates:
                statements = self._parse_xml_to_statements(xml_output)
                if statements:
                    parsed_candidates.append(statements)

            all_candidates.extend(parsed_candidates)

            # Check if we have enough statements
            total_stmts = sum(len(c) for c in parsed_candidates)
            logger.info(f"Attempt {attempt + 1}/{options.max_attempts}: {len(parsed_candidates)} beams, {total_stmts} total statements")

            if total_stmts >= min_expected:
                break

        if not all_candidates:
            return []

        # Select or merge beams
        if options.merge_beams:
            statements = beam_scorer.merge_beams(all_candidates, original_text)
        else:
            statements = beam_scorer.select_best_beam(all_candidates, original_text)

        # Apply embedding-based deduplication if enabled
        if options.embedding_dedup and options.deduplicate:
            try:
                comparer = self._get_predicate_comparer(options)
                if comparer:
                    statements = comparer.deduplicate_statements(
                        statements,
                        entity_canonicalizer=options.entity_canonicalizer
                    )
                    # Also normalize predicates if taxonomy provided
                    if options.predicate_taxonomy or self._predicate_taxonomy:
                        statements = comparer.normalize_predicates(statements)
            except Exception as e:
                logger.warning(f"Embedding deduplication failed, falling back to exact match: {e}")
                statements = self._deduplicate_statements_exact(statements, options)
        elif options.deduplicate:
            statements = self._deduplicate_statements_exact(statements, options)

        return statements

    def _generate_candidate_beams(
        self,
        inputs,
        options: ExtractionOptions,
    ) -> list[str]:
        """Generate multiple candidate beams using diverse beam search."""
        num_seqs = options.num_beams

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=options.max_new_tokens,
                num_beams=num_seqs,
                num_beam_groups=num_seqs,
                num_return_sequences=num_seqs,
                diversity_penalty=options.diversity_penalty,
                do_sample=False,
                trust_remote_code=True,
            )

        # Decode and process candidates
        end_tag = "</statements>"
        candidates: list[str] = []

        for output in outputs:
            decoded = self.tokenizer.decode(output, skip_special_tokens=True)

            # Truncate at </statements>
            if end_tag in decoded:
                end_pos = decoded.find(end_tag) + len(end_tag)
                decoded = decoded[:end_pos]
                candidates.append(decoded)

        # Include fallback if no valid candidates
        if not candidates and len(outputs) > 0:
            fallback = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            candidates.append(fallback)

        return candidates

    def _extract_raw_xml(
        self,
        text: str,
        options: ExtractionOptions,
    ) -> str:
        """
        Extract and return raw XML (legacy method for backward compatibility).

        Uses length-based selection like the original implementation.
        """
        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=4096,
            truncation=True,
        ).to(self.device)

        num_sentences = self._count_sentences(text)
        min_expected = int(num_sentences * options.min_statement_ratio)

        all_results: list[tuple[str, int]] = []

        for attempt in range(options.max_attempts):
            candidates = self._generate_candidate_beams(inputs, options)

            for candidate in candidates:
                if options.deduplicate:
                    candidate = self._deduplicate_xml(candidate)
                num_stmts = self._count_statements(candidate)
                all_results.append((candidate, num_stmts))

            best_so_far = max(all_results, key=lambda x: x[1])[1] if all_results else 0
            if best_so_far >= min_expected:
                break

        if not all_results:
            return "<statements></statements>"

        # Select best result (longest, for backward compatibility)
        return max(all_results, key=lambda x: len(x[0]))[0]

    def _deduplicate_statements_exact(
        self,
        statements: list[Statement],
        options: ExtractionOptions,
    ) -> list[Statement]:
        """Deduplicate statements using exact text matching."""
        from .canonicalization import deduplicate_statements_exact
        return deduplicate_statements_exact(
            statements,
            entity_canonicalizer=options.entity_canonicalizer
        )

    def _deduplicate_xml(self, xml_output: str) -> str:
        """Remove duplicate <stmt> blocks from XML output (legacy method)."""
        try:
            root = ET.fromstring(xml_output)
        except ET.ParseError:
            return xml_output

        if root.tag != 'statements':
            return xml_output

        seen: set[tuple[str, str, str]] = set()
        unique_stmts: list[ET.Element] = []

        for stmt in root.findall('stmt'):
            subject = stmt.findtext('subject', '').strip().lower()
            predicate = stmt.findtext('predicate', '').strip().lower()
            obj = stmt.findtext('object', '').strip().lower()
            key = (subject, predicate, obj)

            if key not in seen:
                seen.add(key)
                unique_stmts.append(stmt)

        new_root = ET.Element('statements')
        for stmt in unique_stmts:
            new_root.append(stmt)

        return ET.tostring(new_root, encoding='unicode')

    def _parse_xml_to_statements(self, xml_output: str) -> list[Statement]:
        """Parse XML output into Statement objects."""
        statements: list[Statement] = []

        try:
            root = ET.fromstring(xml_output)
        except ET.ParseError:
            logger.warning("Failed to parse XML output")
            return statements

        if root.tag != 'statements':
            return statements

        for stmt_elem in root.findall('stmt'):
            try:
                # Parse subject
                subject_elem = stmt_elem.find('subject')
                subject_text = subject_elem.text.strip() if subject_elem is not None and subject_elem.text else ""
                subject_type = self._parse_entity_type(subject_elem.get('type') if subject_elem is not None else None)

                # Parse object
                object_elem = stmt_elem.find('object')
                object_text = object_elem.text.strip() if object_elem is not None and object_elem.text else ""
                object_type = self._parse_entity_type(object_elem.get('type') if object_elem is not None else None)

                # Parse predicate
                predicate_elem = stmt_elem.find('predicate')
                predicate = predicate_elem.text.strip() if predicate_elem is not None and predicate_elem.text else ""

                # Parse source text
                text_elem = stmt_elem.find('text')
                source_text = text_elem.text.strip() if text_elem is not None and text_elem.text else None

                if subject_text and predicate and object_text:
                    statements.append(Statement(
                        subject=Entity(text=subject_text, type=subject_type),
                        predicate=predicate,
                        object=Entity(text=object_text, type=object_type),
                        source_text=source_text,
                    ))
            except Exception as e:
                logger.warning(f"Failed to parse statement: {e}")
                continue

        return statements

    def _parse_entity_type(self, type_str: Optional[str]) -> EntityType:
        """Parse entity type string to EntityType enum."""
        if type_str is None:
            return EntityType.UNKNOWN
        try:
            return EntityType(type_str.upper())
        except ValueError:
            return EntityType.UNKNOWN

    @staticmethod
    def _count_sentences(text: str) -> int:
        """Count approximate number of sentences in text."""
        clean_text = re.sub(r'<[^>]+>', '', text)
        sentences = re.split(r'[.!?]+', clean_text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        return max(1, len(sentences))

    @staticmethod
    def _count_statements(xml_output: str) -> int:
        """Count number of <stmt> tags in output."""
        return len(re.findall(r'<stmt>', xml_output))


# Convenience functions for simple usage

_default_extractor: Optional[StatementExtractor] = None


def _get_default_extractor() -> StatementExtractor:
    """Get or create the default extractor instance."""
    global _default_extractor
    if _default_extractor is None:
        _default_extractor = StatementExtractor()
    return _default_extractor


def extract_statements(
    text: str,
    options: Optional[ExtractionOptions] = None,
    **kwargs,
) -> ExtractionResult:
    """
    Extract structured statements from text.

    This is a convenience function that uses a default StatementExtractor instance.
    For more control, create your own StatementExtractor.

    By default, uses embedding-based deduplication and beam merging for
    high-quality extraction. Requires sentence-transformers package.

    Args:
        text: Input text to extract statements from
        options: Extraction options (or pass individual options as kwargs)
        **kwargs: Individual option overrides (num_beams, diversity_penalty, etc.)

    Returns:
        ExtractionResult containing Statement objects

    Example:
        >>> result = extract_statements("Apple announced a new product.")
        >>> for stmt in result:
        ...     print(f"{stmt.subject.text} -> {stmt.predicate} -> {stmt.object.text}")
    """
    if options is None and kwargs:
        options = ExtractionOptions(**kwargs)
    return _get_default_extractor().extract(text, options)


def extract_statements_as_xml(
    text: str,
    options: Optional[ExtractionOptions] = None,
    **kwargs,
) -> str:
    """
    Extract statements and return raw XML output.

    Args:
        text: Input text to extract statements from
        options: Extraction options
        **kwargs: Individual option overrides

    Returns:
        XML string with <statements> containing <stmt> elements
    """
    if options is None and kwargs:
        options = ExtractionOptions(**kwargs)
    return _get_default_extractor().extract_as_xml(text, options)


def extract_statements_as_json(
    text: str,
    options: Optional[ExtractionOptions] = None,
    indent: Optional[int] = 2,
    **kwargs,
) -> str:
    """
    Extract statements and return JSON string.

    Args:
        text: Input text to extract statements from
        options: Extraction options
        indent: JSON indentation (None for compact)
        **kwargs: Individual option overrides

    Returns:
        JSON string representation of the extraction result
    """
    if options is None and kwargs:
        options = ExtractionOptions(**kwargs)
    return _get_default_extractor().extract_as_json(text, options, indent)


def extract_statements_as_dict(
    text: str,
    options: Optional[ExtractionOptions] = None,
    **kwargs,
) -> dict:
    """
    Extract statements and return as dictionary.

    Args:
        text: Input text to extract statements from
        options: Extraction options
        **kwargs: Individual option overrides

    Returns:
        Dictionary representation of the extraction result
    """
    if options is None and kwargs:
        options = ExtractionOptions(**kwargs)
    return _get_default_extractor().extract_as_dict(text, options)

"""
Statement Extractor - Extract structured statements from text using T5-Gemma 2.

This module uses Diverse Beam Search (Vijayakumar et al., 2016) to generate
multiple candidate extractions and selects the best result.

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
    ):
        """
        Initialize the statement extractor.

        Args:
            model_id: HuggingFace model ID or local path
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
            torch_dtype: Torch dtype (default: bfloat16 on GPU, float32 on CPU)
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

        # Wrap text in page tags if not already wrapped
        if not text.startswith("<page>"):
            text = f"<page>{text}</page>"

        # Run extraction with retry logic
        xml_output = self._extract_with_retry(text, options)

        # Parse XML to statements
        statements = self._parse_xml_to_statements(xml_output)

        return ExtractionResult(
            statements=statements,
            source_text=text,
        )

    def extract_as_xml(
        self,
        text: str,
        options: Optional[ExtractionOptions] = None,
    ) -> str:
        """
        Extract statements and return raw XML output.

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

        return self._extract_with_retry(text, options)

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

    def _extract_with_retry(
        self,
        text: str,
        options: ExtractionOptions,
    ) -> str:
        """Run extraction with retry logic for under-extraction."""
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

        all_results: list[tuple[str, int]] = []

        for attempt in range(options.max_attempts):
            result = self._run_single_extraction(inputs, options)
            num_stmts = self._count_statements(result)
            all_results.append((result, num_stmts))

            logger.info(f"Attempt {attempt + 1}/{options.max_attempts}: {num_stmts} statements")

            if num_stmts >= min_expected:
                break

        # Select best result (longest, which typically has most statements)
        return max(all_results, key=lambda x: len(x[0]))[0]

    def _run_single_extraction(
        self,
        inputs,
        options: ExtractionOptions,
    ) -> str:
        """Run a single extraction attempt using diverse beam search."""
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

                if options.deduplicate:
                    decoded = self._deduplicate_xml(decoded)

                candidates.append(decoded)

        if candidates:
            return max(candidates, key=len)
        else:
            # Fallback to first output
            fallback = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            if options.deduplicate and '<stmt>' in fallback:
                fallback = self._deduplicate_xml(fallback)
            return fallback

    def _deduplicate_xml(self, xml_output: str) -> str:
        """Remove duplicate <stmt> blocks from XML output."""
        try:
            root = ET.fromstring(xml_output)
        except ET.ParseError:
            return xml_output

        if root.tag != 'statements':
            return xml_output

        seen: set[tuple[str, str, str]] = set()
        unique_stmts: list[ET.Element] = []

        for stmt in root.findall('stmt'):
            subject = stmt.findtext('subject', '').strip()
            predicate = stmt.findtext('predicate', '').strip()
            obj = stmt.findtext('object', '').strip()
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
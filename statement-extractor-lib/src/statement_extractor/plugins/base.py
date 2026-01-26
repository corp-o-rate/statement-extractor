"""
Base plugin classes for the extraction pipeline.

Defines the abstract interfaces for each pipeline stage:
- BaseSplitterPlugin: Stage 1 - Text → SplitSentence (atomic sentences)
- BaseExtractorPlugin: Stage 2 - SplitSentence → PipelineStatement (triples)
- BaseQualifierPlugin: Stage 3 - Entity → CanonicalEntity
- BaseLabelerPlugin: Stage 4 - Statement → StatementLabel
- BaseTaxonomyPlugin: Stage 5 - Statement → TaxonomyResult

Content acquisition plugins (for URL processing):
- BaseScraperPlugin: Fetch content from URLs
- BasePDFParserPlugin: Extract text from PDFs
"""

from abc import ABC, abstractmethod
from enum import Enum, Flag, auto
from typing import TYPE_CHECKING, Any, Optional

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from ..pipeline.context import PipelineContext
    from ..models import (
        SplitSentence,
        PipelineStatement,
        ExtractedEntity,
        CanonicalEntity,
        StatementLabel,
        TaxonomyResult,
        EntityType,
    )


class PluginCapability(Flag):
    """Flags indicating plugin capabilities."""
    NONE = 0
    BATCH_PROCESSING = auto()   # Can process multiple items at once
    ASYNC_PROCESSING = auto()   # Supports async execution
    EXTERNAL_API = auto()       # Uses external API (may have rate limits)
    LLM_REQUIRED = auto()       # Requires an LLM model
    CACHING = auto()            # Supports result caching


def get_available_vram_gb() -> float:
    """
    Get available GPU VRAM in GB.

    Returns 0.0 if no GPU is available or VRAM cannot be determined.
    """
    try:
        import torch
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            total = torch.cuda.get_device_properties(device).total_memory
            allocated = torch.cuda.memory_allocated(device)
            available = (total - allocated) / (1024 ** 3)  # Convert to GB
            return available
        elif torch.backends.mps.is_available():
            # MPS doesn't expose VRAM info; estimate based on typical Apple Silicon
            # Return a conservative estimate
            return 8.0
    except ImportError:
        pass
    return 0.0


def calculate_batch_size(
    per_item_vram_gb: float,
    overhead_gb: float = 2.0,
    min_batch: int = 1,
    max_batch: int = 32,
) -> int:
    """
    Calculate optimal batch size based on available VRAM.

    Args:
        per_item_vram_gb: VRAM required per item in GB
        overhead_gb: Reserved VRAM for model weights and system overhead
        min_batch: Minimum batch size
        max_batch: Maximum batch size cap

    Returns:
        Optimal batch size for the current GPU
    """
    available = get_available_vram_gb()
    if available <= 0 or per_item_vram_gb <= 0:
        return min_batch

    usable = max(0, available - overhead_gb)
    batch_size = int(usable / per_item_vram_gb)
    return max(min_batch, min(batch_size, max_batch))


class BasePlugin(ABC):
    """
    Base class for all pipeline plugins.

    All plugins must implement the name property and can optionally
    override priority and capabilities.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name for this plugin (used for registration and CLI)."""
        ...

    @property
    def priority(self) -> int:
        """
        Plugin priority (lower = higher priority, runs first).

        Default is 100. Use lower values (e.g., 10, 20) for critical plugins
        that should run before others.
        """
        return 100

    @property
    def capabilities(self) -> PluginCapability:
        """Plugin capabilities (flags)."""
        return PluginCapability.NONE

    @property
    def description(self) -> str:
        """Human-readable description of this plugin."""
        return ""

    @property
    def model_vram_gb(self) -> float:
        """
        Estimated VRAM required for model weights in GB.

        Override this if the plugin loads a GPU model. This is used to
        reserve memory overhead when calculating batch sizes.

        Default is 0.0 (no GPU model).
        """
        return 0.0

    @property
    def per_item_vram_gb(self) -> float:
        """
        Estimated VRAM required per item during batch processing in GB.

        Override this for plugins with BATCH_PROCESSING capability.
        Used to calculate optimal batch size: batch = (available - overhead) / per_item

        Default is 0.1 GB (100MB) as a conservative estimate.
        """
        return 0.1

    def get_optimal_batch_size(self, max_batch: int = 32) -> int:
        """
        Calculate optimal batch size based on available VRAM and plugin requirements.

        Args:
            max_batch: Maximum batch size cap

        Returns:
            Optimal batch size for current GPU state
        """
        if not (PluginCapability.BATCH_PROCESSING in self.capabilities):
            return 1

        return calculate_batch_size(
            per_item_vram_gb=self.per_item_vram_gb,
            overhead_gb=self.model_vram_gb + 1.0,  # Add 1GB system overhead
            min_batch=1,
            max_batch=max_batch,
        )


class BaseSplitterPlugin(BasePlugin):
    """
    Stage 1 plugin: Split text into atomic sentences.

    Takes raw text and produces SplitSentence objects containing
    atomic statements that can be converted to triples in Stage 2.
    """

    @abstractmethod
    def split(
        self,
        text: str,
        context: "PipelineContext",
    ) -> list["SplitSentence"]:
        """
        Split text into atomic sentences.

        Args:
            text: Input text to split
            context: Pipeline context for accessing metadata and config

        Returns:
            List of SplitSentence objects
        """
        ...

    def split_batch(
        self,
        texts: list[str],
        context: "PipelineContext",
    ) -> list[list["SplitSentence"]]:
        """
        Split multiple texts into atomic sentences in a single batch.

        Default implementation calls split() for each text sequentially.
        Plugins with BATCH_PROCESSING capability should override this
        for efficient GPU batching using get_optimal_batch_size().

        Args:
            texts: List of input texts to split
            context: Pipeline context for accessing metadata and config

        Returns:
            List of SplitSentence lists, one per input text
        """
        return [self.split(text, context) for text in texts]


class BaseExtractorPlugin(BasePlugin):
    """
    Stage 2 plugin: Extract subject-predicate-object triples from sentences.

    Takes SplitSentence objects and produces PipelineStatement objects
    with ExtractedEntity subjects/objects that have types, spans,
    and confidence scores.
    """

    @abstractmethod
    def extract(
        self,
        split_sentences: list["SplitSentence"],
        context: "PipelineContext",
    ) -> list["PipelineStatement"]:
        """
        Extract triples from split sentences.

        Args:
            split_sentences: Atomic sentences from Stage 1
            context: Pipeline context

        Returns:
            List of PipelineStatement objects with typed entities
        """
        ...


class BaseQualifierPlugin(BasePlugin):
    """
    Stage 3 plugin: Qualify entities with identifiers and canonical forms.

    Processes entities of specific types and adds:
    - Semantic qualifiers (role, org for PERSON entities)
    - External identifiers (LEI, company number, SEC CIK)
    - Canonical name and FQN (fully qualified name)

    Returns a CanonicalEntity ready for use in labeled statements.
    """

    @property
    @abstractmethod
    def supported_entity_types(self) -> set["EntityType"]:
        """Entity types this plugin can qualify (e.g., {ORG, PERSON})."""
        ...

    @property
    def supported_identifier_types(self) -> list[str]:
        """
        Identifier types this plugin can use for lookup.

        For example, GLEIFQualifier can lookup by 'lei'.
        """
        return []

    @property
    def provided_identifier_types(self) -> list[str]:
        """
        Identifier types this plugin can provide.

        For example, GLEIFQualifier provides 'lei', 'jurisdiction'.
        """
        return []

    @abstractmethod
    def qualify(
        self,
        entity: "ExtractedEntity",
        context: "PipelineContext",
    ) -> "CanonicalEntity | None":
        """
        Qualify an entity and return its canonical form.

        This method should:
        1. Look up identifiers (LEI, CIK, company number, etc.)
        2. Find the canonical name if available
        3. Generate the FQN (fully qualified name)
        4. Return a CanonicalEntity with all information

        Args:
            entity: The entity to qualify
            context: Pipeline context (for accessing source text, other entities)

        Returns:
            CanonicalEntity with qualifiers and FQN, or None if entity not found
        """
        ...


class ClassificationSchema:
    """
    Schema for simple multi-choice classification (2-20 choices).

    Handled by GLiNER2 `.classification()` in a single pass.

    Examples:
        - sentiment: ["positive", "negative", "neutral"]
        - certainty: ["certain", "uncertain", "speculative"]
        - temporality: ["past", "present", "future"]
    """

    def __init__(
        self,
        label_type: str,
        choices: list[str],
        description: str = "",
        scope: str = "statement",  # "statement", "subject", "object", "predicate"
    ):
        self.label_type = label_type
        self.choices = choices
        self.description = description
        self.scope = scope

    def __repr__(self) -> str:
        return f"ClassificationSchema({self.label_type!r}, choices={self.choices!r})"


class TaxonomySchema:
    """
    Schema for large taxonomy labeling (100s of values).

    Too many choices for GLiNER2 classification. Requires MNLI or similar:
    - MNLI zero-shot with label descriptions
    - Embedding-based nearest neighbor search
    - Hierarchical classification (category → subcategory)

    Examples:
        - industry_code: NAICS/SIC codes (1000+ values)
        - relation_type: detailed relation ontology (100+ types)
        - job_title: standardized job taxonomy
    """

    def __init__(
        self,
        label_type: str,
        values: list[str] | dict[str, list[str]],  # flat list or hierarchical dict
        description: str = "",
        scope: str = "statement",  # "statement", "subject", "object", "predicate"
        label_descriptions: dict[str, str] | None = None,  # descriptions for MNLI
    ):
        self.label_type = label_type
        self.values = values
        self.description = description
        self.scope = scope
        self.label_descriptions = label_descriptions  # e.g., {"NAICS:5112": "Software Publishers"}

    @property
    def is_hierarchical(self) -> bool:
        """Check if taxonomy is hierarchical (dict) vs flat (list)."""
        return isinstance(self.values, dict)

    @property
    def all_values(self) -> list[str]:
        """Get all taxonomy values (flattened if hierarchical)."""
        if isinstance(self.values, list):
            return self.values
        # Flatten hierarchical dict
        result = []
        for category, subcategories in self.values.items():
            result.append(category)
            result.extend(subcategories)
        return result

    def __repr__(self) -> str:
        count = len(self.all_values)
        return f"TaxonomySchema({self.label_type!r}, {count} values)"


class BaseLabelerPlugin(BasePlugin):
    """
    Stage 4 plugin: Apply labels to statements.

    Adds classification labels (sentiment, relation type, confidence)
    to the final labeled statements.

    Labelers can provide a classification_schema that extractors will use
    to run classification in a single model pass. The results are stored
    in the pipeline context for the labeler to retrieve.
    """

    @property
    @abstractmethod
    def label_type(self) -> str:
        """
        The type of label this plugin produces.

        Examples: 'sentiment', 'relation_type', 'confidence'
        """
        ...

    @property
    def classification_schema(self) -> ClassificationSchema | None:
        """
        Simple multi-choice classification schema (2-20 choices).

        If provided, GLiNER2 extractor will run `.classification()` and store
        results in context for this labeler to retrieve.

        Returns:
            ClassificationSchema or None
        """
        return None

    @property
    def taxonomy_schema(self) -> TaxonomySchema | None:
        """
        Large taxonomy schema (100s of values).

        If provided, requires MNLI or embedding-based classification.
        Results stored in context for this labeler to retrieve.

        Returns:
            TaxonomySchema or None
        """
        return None

    @abstractmethod
    def label(
        self,
        statement: "PipelineStatement",
        subject_canonical: "CanonicalEntity",
        object_canonical: "CanonicalEntity",
        context: "PipelineContext",
    ) -> "StatementLabel | None":
        """
        Apply a label to a statement.

        Args:
            statement: The statement to label
            subject_canonical: Canonicalized subject entity
            object_canonical: Canonicalized object entity
            context: Pipeline context (check context.classification_results for pre-computed labels)

        Returns:
            StatementLabel if applicable, None otherwise
        """
        ...


class BaseTaxonomyPlugin(BasePlugin):
    """
    Stage 5 plugin: Classify statements against a taxonomy.

    Taxonomy classification is separate from labeling because:
    - It operates on large taxonomies (100s-1000s of labels)
    - It requires specialized models (MNLI, embeddings)
    - It's computationally heavier than simple labeling

    Taxonomy plugins produce TaxonomyResult objects that are stored
    in the pipeline context.
    """

    @property
    @abstractmethod
    def taxonomy_name(self) -> str:
        """
        Name of the taxonomy this plugin classifies against.

        Examples: 'esg_topics', 'industry_codes', 'relation_types'
        """
        ...

    @property
    def taxonomy_schema(self) -> TaxonomySchema | None:
        """
        The taxonomy schema this plugin uses.

        Returns:
            TaxonomySchema describing the taxonomy structure
        """
        return None

    @property
    def supported_categories(self) -> list[str]:
        """
        List of taxonomy categories this plugin supports.

        Returns empty list if all categories are supported.
        """
        return []

    @abstractmethod
    def classify(
        self,
        statement: "PipelineStatement",
        subject_canonical: "CanonicalEntity",
        object_canonical: "CanonicalEntity",
        context: "PipelineContext",
    ) -> list["TaxonomyResult"]:
        """
        Classify a statement against the taxonomy.

        Returns all labels above the confidence threshold. A single statement
        may have multiple applicable taxonomy labels.

        Args:
            statement: The statement to classify
            subject_canonical: Canonicalized subject entity
            object_canonical: Canonicalized object entity
            context: Pipeline context

        Returns:
            List of TaxonomyResult objects (empty if none above threshold)
        """
        ...

    def classify_batch(
        self,
        items: list[tuple["PipelineStatement", "CanonicalEntity", "CanonicalEntity"]],
        context: "PipelineContext",
    ) -> list[list["TaxonomyResult"]]:
        """
        Classify multiple statements against the taxonomy in a single batch.

        Default implementation calls classify() for each statement sequentially.
        Plugins with BATCH_PROCESSING capability should override this
        for efficient GPU batching using get_optimal_batch_size().

        Args:
            items: List of (statement, subject_canonical, object_canonical) tuples
            context: Pipeline context

        Returns:
            List of TaxonomyResult lists, one per input statement
        """
        return [
            self.classify(stmt, subj, obj, context)
            for stmt, subj, obj in items
        ]


# =============================================================================
# Content Acquisition Plugins (for URL processing)
# =============================================================================


class ContentType(str, Enum):
    """Content type detected from URL or HTTP response."""
    HTML = "html"
    PDF = "pdf"
    BINARY = "binary"
    UNKNOWN = "unknown"


class ScraperResult(BaseModel):
    """Result from a scraper plugin."""
    url: str = Field(description="Original URL requested")
    final_url: str = Field(description="Final URL after redirects")
    content: bytes = Field(description="Raw content bytes")
    content_type: ContentType = Field(description="Detected content type")
    headers: dict[str, str] = Field(default_factory=dict, description="Response headers")
    error: Optional[str] = Field(default=None, description="Error message if fetch failed")

    model_config = {"arbitrary_types_allowed": True}

    @property
    def ok(self) -> bool:
        """Check if the fetch was successful."""
        return self.error is None and len(self.content) > 0


class PDFParseResult(BaseModel):
    """Result from a PDF parser plugin."""
    pages: list[str] = Field(description="Extracted text for each page")
    page_count: int = Field(description="Total number of pages in PDF")
    metadata: dict[str, Any] = Field(default_factory=dict, description="PDF metadata (title, author, etc)")
    error: Optional[str] = Field(default=None, description="Error message if parsing failed")

    @property
    def ok(self) -> bool:
        """Check if parsing was successful."""
        return self.error is None

    @property
    def full_text(self) -> str:
        """Get concatenated text from all pages."""
        return "\n\n".join(self.pages)


class BaseScraperPlugin(BasePlugin):
    """
    Plugin for fetching content from URLs.

    Scrapers handle HTTP requests, redirects, retries, and content type detection.
    They return raw bytes that can be processed by appropriate parsers (HTML, PDF, etc).

    Example implementation:
        @PluginRegistry.scraper
        class MyScraperPlugin(BaseScraperPlugin):
            @property
            def name(self) -> str:
                return "my_scraper"

            async def fetch(self, url: str, timeout: float = 30.0) -> ScraperResult:
                # Implement fetching logic
                ...
    """

    @property
    def capabilities(self) -> PluginCapability:
        """Scrapers support async processing by default."""
        return PluginCapability.ASYNC_PROCESSING | PluginCapability.EXTERNAL_API

    @abstractmethod
    async def fetch(self, url: str, timeout: float = 30.0) -> ScraperResult:
        """
        Fetch content from a URL.

        Args:
            url: The URL to fetch
            timeout: Request timeout in seconds

        Returns:
            ScraperResult with content, content type, and any errors
        """
        ...

    async def head(self, url: str, timeout: float = 10.0) -> ScraperResult:
        """
        Check content type without downloading the full body.

        Default implementation does a full fetch. Override for efficiency.

        Args:
            url: The URL to check
            timeout: Request timeout in seconds

        Returns:
            ScraperResult with content_type populated (content may be empty)
        """
        return await self.fetch(url, timeout)

    def is_supported_url(self, url: str) -> bool:
        """
        Check if this scraper can handle the URL.

        Override to restrict to specific URL patterns or domains.

        Args:
            url: The URL to check

        Returns:
            True if this scraper can handle the URL
        """
        return True


class BasePDFParserPlugin(BasePlugin):
    """
    Plugin for extracting text from PDF files.

    PDF parsers take raw PDF bytes and extract text content page by page.
    They may support OCR for image-heavy PDFs.

    Example implementation:
        @PluginRegistry.pdf_parser
        class MyPDFParserPlugin(BasePDFParserPlugin):
            @property
            def name(self) -> str:
                return "my_pdf_parser"

            def parse(self, pdf_bytes: bytes, ...) -> PDFParseResult:
                # Implement parsing logic
                ...
    """

    @abstractmethod
    def parse(
        self,
        pdf_bytes: bytes,
        max_pages: int = 500,
        use_ocr: bool = False,
    ) -> PDFParseResult:
        """
        Extract text from PDF bytes.

        Args:
            pdf_bytes: Raw PDF file content
            max_pages: Maximum number of pages to process
            use_ocr: Force OCR even for text-extractable PDFs

        Returns:
            PDFParseResult with extracted text for each page
        """
        ...

    @property
    def supports_ocr(self) -> bool:
        """
        Whether this parser supports OCR for image-heavy PDFs.

        Returns:
            True if OCR is available
        """
        return False

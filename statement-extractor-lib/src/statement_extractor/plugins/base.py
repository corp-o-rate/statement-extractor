"""
Base plugin classes for the extraction pipeline.

Defines the abstract interfaces for each pipeline stage:
- BaseSplitterPlugin: Stage 1 - Text → RawTriple
- BaseExtractorPlugin: Stage 2 - RawTriple → PipelineStatement
- BaseQualifierPlugin: Stage 3 - Entity → EntityQualifiers
- BaseCanonicalizerPlugin: Stage 4 - QualifiedEntity → CanonicalMatch
- BaseLabelerPlugin: Stage 5 - Statement → StatementLabel
"""

from abc import ABC, abstractmethod
from enum import Flag, auto
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..pipeline.context import PipelineContext
    from ..models import (
        RawTriple,
        PipelineStatement,
        ExtractedEntity,
        EntityQualifiers,
        QualifiedEntity,
        CanonicalMatch,
        CanonicalEntity,
        StatementLabel,
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


class BaseSplitterPlugin(BasePlugin):
    """
    Stage 1 plugin: Split text into atomic triples.

    Takes raw text and produces RawTriple objects containing
    subject/predicate/object text and source sentence.
    """

    @abstractmethod
    def split(
        self,
        text: str,
        context: "PipelineContext",
    ) -> list["RawTriple"]:
        """
        Split text into atomic triples.

        Args:
            text: Input text to split
            context: Pipeline context for accessing metadata and config

        Returns:
            List of RawTriple objects
        """
        ...


class BaseExtractorPlugin(BasePlugin):
    """
    Stage 2 plugin: Refine triples into statements with typed entities.

    Takes RawTriple objects and produces PipelineStatement objects
    with ExtractedEntity subjects/objects that have types, spans,
    and confidence scores.
    """

    @abstractmethod
    def extract(
        self,
        raw_triples: list["RawTriple"],
        context: "PipelineContext",
    ) -> list["PipelineStatement"]:
        """
        Extract statements from raw triples.

        Args:
            raw_triples: Raw triples from Stage 1
            context: Pipeline context

        Returns:
            List of PipelineStatement objects with typed entities
        """
        ...


class BaseQualifierPlugin(BasePlugin):
    """
    Stage 3 plugin: Add qualifiers and identifiers to entities.

    Processes entities of specific types and adds semantic qualifiers
    (role, org) or external identifiers (LEI, company number).
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
    ) -> "EntityQualifiers | None":
        """
        Add qualifiers to an entity.

        Args:
            entity: The entity to qualify
            context: Pipeline context (for accessing source text, other entities)

        Returns:
            EntityQualifiers with added information, or None if nothing to add
        """
        ...


class BaseCanonicalizerPlugin(BasePlugin):
    """
    Stage 4 plugin: Resolve entities to canonical forms.

    Takes qualified entities and finds their canonical representations
    using various matching strategies (identifier, name, fuzzy, LLM).
    """

    @property
    @abstractmethod
    def supported_entity_types(self) -> set["EntityType"]:
        """Entity types this plugin can canonicalize."""
        ...

    @abstractmethod
    def find_canonical(
        self,
        entity: "QualifiedEntity",
        context: "PipelineContext",
    ) -> "CanonicalMatch | None":
        """
        Find canonical form for an entity.

        Args:
            entity: Qualified entity to canonicalize
            context: Pipeline context

        Returns:
            CanonicalMatch if found, None otherwise
        """
        ...

    def format_fqn(
        self,
        entity: "QualifiedEntity",
        match: "CanonicalMatch | None",
    ) -> str:
        """
        Format the fully qualified name for display.

        Can be overridden by subclasses for custom formatting.
        Default implementation uses CanonicalEntity._generate_fqn.
        """
        from ..models import CanonicalEntity
        return CanonicalEntity._generate_fqn(entity, match)


class BaseLabelerPlugin(BasePlugin):
    """
    Stage 5 plugin: Apply labels to statements.

    Adds classification labels (sentiment, relation type, confidence)
    to the final labeled statements.
    """

    @property
    @abstractmethod
    def label_type(self) -> str:
        """
        The type of label this plugin produces.

        Examples: 'sentiment', 'relation_type', 'confidence'
        """
        ...

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
            context: Pipeline context

        Returns:
            StatementLabel if applicable, None otherwise
        """
        ...

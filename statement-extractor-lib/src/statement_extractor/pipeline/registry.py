"""
PluginRegistry - Registration and discovery of plugins.

Provides a central registry for all plugin types with decorator-based
registration and discovery by entity type.
"""

import logging
from typing import TYPE_CHECKING, Type, TypeVar

if TYPE_CHECKING:
    from ..plugins.base import (
        BasePlugin,
        BaseSplitterPlugin,
        BaseExtractorPlugin,
        BaseQualifierPlugin,
        BaseLabelerPlugin,
        BaseTaxonomyPlugin,
        BaseScraperPlugin,
        BasePDFParserPlugin,
    )
    from ..models import EntityType

logger = logging.getLogger(__name__)

T = TypeVar("T", bound="BasePlugin")


class PluginRegistry:
    """
    Central registry for all pipeline plugins.

    Supports registration by decorator or explicit method call.
    Plugins are sorted by priority (lower = higher priority).
    """

    # Class-level storage for registered plugins
    _splitters: list["BaseSplitterPlugin"] = []
    _extractors: list["BaseExtractorPlugin"] = []
    _qualifiers: list["BaseQualifierPlugin"] = []
    _labelers: list["BaseLabelerPlugin"] = []
    _taxonomy_classifiers: list["BaseTaxonomyPlugin"] = []

    # Content acquisition plugins
    _scrapers: list["BaseScraperPlugin"] = []
    _pdf_parsers: list["BasePDFParserPlugin"] = []

    # Index by entity type for quick lookup
    _qualifiers_by_type: dict["EntityType", list["BaseQualifierPlugin"]] = {}

    # Index by name for CLI lookup
    _all_plugins: dict[str, "BasePlugin"] = {}

    @classmethod
    def clear(cls) -> None:
        """Clear all registered plugins (useful for testing)."""
        cls._splitters = []
        cls._extractors = []
        cls._qualifiers = []
        cls._labelers = []
        cls._taxonomy_classifiers = []
        cls._scrapers = []
        cls._pdf_parsers = []
        cls._qualifiers_by_type = {}
        cls._all_plugins = {}

    # =========================================================================
    # Registration methods
    # =========================================================================

    @classmethod
    def register_splitter(cls, plugin: "BaseSplitterPlugin") -> None:
        """Register a splitter plugin."""
        cls._splitters.append(plugin)
        cls._splitters.sort(key=lambda p: p.priority)
        cls._all_plugins[plugin.name] = plugin
        logger.debug(f"Registered splitter: {plugin.name} (priority={plugin.priority})")

    @classmethod
    def register_extractor(cls, plugin: "BaseExtractorPlugin") -> None:
        """Register an extractor plugin."""
        cls._extractors.append(plugin)
        cls._extractors.sort(key=lambda p: p.priority)
        cls._all_plugins[plugin.name] = plugin
        logger.debug(f"Registered extractor: {plugin.name} (priority={plugin.priority})")

    @classmethod
    def register_qualifier(cls, plugin: "BaseQualifierPlugin") -> None:
        """Register a qualifier plugin."""
        cls._qualifiers.append(plugin)
        cls._qualifiers.sort(key=lambda p: p.priority)
        cls._all_plugins[plugin.name] = plugin

        # Index by entity type
        for entity_type in plugin.supported_entity_types:
            if entity_type not in cls._qualifiers_by_type:
                cls._qualifiers_by_type[entity_type] = []
            cls._qualifiers_by_type[entity_type].append(plugin)
            cls._qualifiers_by_type[entity_type].sort(key=lambda p: p.priority)

        logger.debug(
            f"Registered qualifier: {plugin.name} "
            f"(priority={plugin.priority}, types={[t.value for t in plugin.supported_entity_types]})"
        )

    @classmethod
    def register_labeler(cls, plugin: "BaseLabelerPlugin") -> None:
        """Register a labeler plugin."""
        cls._labelers.append(plugin)
        cls._labelers.sort(key=lambda p: p.priority)
        cls._all_plugins[plugin.name] = plugin
        logger.debug(f"Registered labeler: {plugin.name} (priority={plugin.priority})")

    @classmethod
    def register_taxonomy(cls, plugin: "BaseTaxonomyPlugin") -> None:
        """Register a taxonomy classifier plugin."""
        cls._taxonomy_classifiers.append(plugin)
        cls._taxonomy_classifiers.sort(key=lambda p: p.priority)
        cls._all_plugins[plugin.name] = plugin
        logger.debug(f"Registered taxonomy: {plugin.name} (priority={plugin.priority})")

    @classmethod
    def register_scraper(cls, plugin: "BaseScraperPlugin") -> None:
        """Register a scraper plugin."""
        cls._scrapers.append(plugin)
        cls._scrapers.sort(key=lambda p: p.priority)
        cls._all_plugins[plugin.name] = plugin
        logger.debug(f"Registered scraper: {plugin.name} (priority={plugin.priority})")

    @classmethod
    def register_pdf_parser(cls, plugin: "BasePDFParserPlugin") -> None:
        """Register a PDF parser plugin."""
        cls._pdf_parsers.append(plugin)
        cls._pdf_parsers.sort(key=lambda p: p.priority)
        cls._all_plugins[plugin.name] = plugin
        logger.debug(f"Registered PDF parser: {plugin.name} (priority={plugin.priority})")

    # =========================================================================
    # Decorator registration
    # =========================================================================

    @classmethod
    def splitter(cls, plugin_class: Type[T]) -> Type[T]:
        """Decorator to register a splitter plugin class."""
        cls.register_splitter(plugin_class())
        return plugin_class

    @classmethod
    def extractor(cls, plugin_class: Type[T]) -> Type[T]:
        """Decorator to register an extractor plugin class."""
        cls.register_extractor(plugin_class())
        return plugin_class

    @classmethod
    def qualifier(cls, plugin_class: Type[T]) -> Type[T]:
        """Decorator to register a qualifier plugin class."""
        cls.register_qualifier(plugin_class())
        return plugin_class

    @classmethod
    def labeler(cls, plugin_class: Type[T]) -> Type[T]:
        """Decorator to register a labeler plugin class."""
        cls.register_labeler(plugin_class())
        return plugin_class

    @classmethod
    def taxonomy(cls, plugin_class: Type[T]) -> Type[T]:
        """Decorator to register a taxonomy classifier plugin class."""
        cls.register_taxonomy(plugin_class())
        return plugin_class

    @classmethod
    def scraper(cls, plugin_class: Type[T]) -> Type[T]:
        """Decorator to register a scraper plugin class."""
        cls.register_scraper(plugin_class())
        return plugin_class

    @classmethod
    def pdf_parser(cls, plugin_class: Type[T]) -> Type[T]:
        """Decorator to register a PDF parser plugin class."""
        cls.register_pdf_parser(plugin_class())
        return plugin_class

    # =========================================================================
    # Retrieval methods
    # =========================================================================

    @classmethod
    def get_splitters(cls) -> list["BaseSplitterPlugin"]:
        """Get all registered splitter plugins (sorted by priority)."""
        return cls._splitters.copy()

    @classmethod
    def get_extractors(cls) -> list["BaseExtractorPlugin"]:
        """Get all registered extractor plugins (sorted by priority)."""
        return cls._extractors.copy()

    @classmethod
    def get_qualifiers(cls) -> list["BaseQualifierPlugin"]:
        """Get all registered qualifier plugins (sorted by priority)."""
        return cls._qualifiers.copy()

    @classmethod
    def get_qualifiers_for_type(cls, entity_type: "EntityType") -> list["BaseQualifierPlugin"]:
        """Get qualifier plugins that support a specific entity type."""
        return cls._qualifiers_by_type.get(entity_type, []).copy()

    @classmethod
    def get_labelers(cls) -> list["BaseLabelerPlugin"]:
        """Get all registered labeler plugins (sorted by priority)."""
        return cls._labelers.copy()

    @classmethod
    def get_taxonomy_classifiers(cls) -> list["BaseTaxonomyPlugin"]:
        """Get all registered taxonomy classifier plugins (sorted by priority)."""
        return cls._taxonomy_classifiers.copy()

    @classmethod
    def get_scrapers(cls) -> list["BaseScraperPlugin"]:
        """Get all registered scraper plugins (sorted by priority)."""
        return cls._scrapers.copy()

    @classmethod
    def get_pdf_parsers(cls) -> list["BasePDFParserPlugin"]:
        """Get all registered PDF parser plugins (sorted by priority)."""
        return cls._pdf_parsers.copy()

    @classmethod
    def get_plugin(cls, name: str) -> "BasePlugin | None":
        """Get a plugin by name."""
        return cls._all_plugins.get(name)

    @classmethod
    def get_all_plugins(cls) -> dict[str, "BasePlugin"]:
        """Get all registered plugins by name."""
        return cls._all_plugins.copy()

    @classmethod
    def get_plugins_for_stage(cls, stage: int) -> list["BasePlugin"]:
        """Get all plugins for a specific stage number."""
        if stage == 1:
            return cls._splitters.copy()
        elif stage == 2:
            return cls._extractors.copy()
        elif stage == 3:
            return cls._qualifiers.copy()
        elif stage == 4:
            return cls._labelers.copy()
        elif stage == 5:
            return cls._taxonomy_classifiers.copy()
        return []

    # =========================================================================
    # Info methods
    # =========================================================================

    @classmethod
    def list_plugins(cls, stage: int | None = None) -> list[dict]:
        """
        List all plugins with their info.

        Args:
            stage: Optional stage number to filter by

        Returns:
            List of plugin info dicts with name, stage, priority, description
        """
        result = []

        plugins_by_stage = [
            (1, "splitting", cls._splitters),
            (2, "extraction", cls._extractors),
            (3, "qualification", cls._qualifiers),
            (4, "labeling", cls._labelers),
            (5, "taxonomy", cls._taxonomy_classifiers),
            # Content acquisition plugins (stage 0)
            (0, "scraper", cls._scrapers),
            (-1, "pdf_parser", cls._pdf_parsers),
        ]

        for stage_num, stage_name, plugins in plugins_by_stage:
            if stage is not None and stage != stage_num:
                continue
            for plugin in plugins:
                info = {
                    "name": plugin.name,
                    "stage": stage_num,
                    "stage_name": stage_name,
                    "priority": plugin.priority,
                    "capabilities": plugin.capabilities.name if plugin.capabilities else "NONE",
                }
                # Add entity types for qualifiers/canonicalizers
                if hasattr(plugin, "supported_entity_types"):
                    info["entity_types"] = [t.value for t in plugin.supported_entity_types]
                # Add label type for labelers
                if hasattr(plugin, "label_type"):
                    info["label_type"] = plugin.label_type
                # Add taxonomy name for taxonomy classifiers
                if hasattr(plugin, "taxonomy_name"):
                    info["taxonomy_name"] = plugin.taxonomy_name
                result.append(info)

        return result

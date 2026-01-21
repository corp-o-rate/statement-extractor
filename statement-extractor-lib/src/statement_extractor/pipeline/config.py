"""
PipelineConfig - Configuration for stage/plugin selection.

Controls which stages are enabled and which plugins to use.
"""

from typing import Any, Optional

from pydantic import BaseModel, Field


class PipelineConfig(BaseModel):
    """
    Configuration for the extraction pipeline.

    Controls which stages are enabled, which plugins to use,
    and stage-specific options.
    """
    # Stage selection (1=Splitting, 2=Extraction, 3=Qualification, 4=Labeling, 5=Taxonomy)
    enabled_stages: set[int] = Field(
        default={1, 2, 3, 4, 5},
        description="Set of enabled stage numbers (1-5)"
    )

    # Plugin selection
    enabled_plugins: Optional[set[str]] = Field(
        None,
        description="Set of enabled plugin names (None = all enabled)"
    )
    disabled_plugins: set[str] = Field(
        default_factory=lambda: {
            "mnli_taxonomy_classifier",  # Disabled by default - use embedding_taxonomy_classifier instead (faster)
        },
        description="Set of disabled plugin names"
    )

    # Stage-specific options
    splitter_options: dict[str, Any] = Field(
        default_factory=dict,
        description="Options passed to splitter plugins"
    )
    extractor_options: dict[str, Any] = Field(
        default_factory=dict,
        description="Options passed to extractor plugins"
    )
    qualifier_options: dict[str, Any] = Field(
        default_factory=dict,
        description="Options passed to qualifier plugins (includes canonicalizers)"
    )
    labeler_options: dict[str, Any] = Field(
        default_factory=dict,
        description="Options passed to labeler plugins"
    )
    taxonomy_options: dict[str, Any] = Field(
        default_factory=dict,
        description="Options passed to taxonomy plugins"
    )

    # General options
    fail_fast: bool = Field(
        default=True,
        description="Stop processing on first error (otherwise continue and collect errors)"
    )
    parallel_processing: bool = Field(
        default=False,
        description="Enable parallel processing where possible"
    )
    max_statements: Optional[int] = Field(
        None,
        description="Maximum number of statements to process (None = unlimited)"
    )

    def is_stage_enabled(self, stage: int) -> bool:
        """Check if a stage is enabled."""
        return stage in self.enabled_stages

    def is_plugin_enabled(self, plugin_name: str) -> bool:
        """Check if a plugin is enabled."""
        if plugin_name in self.disabled_plugins:
            return False
        if self.enabled_plugins is None:
            return True
        return plugin_name in self.enabled_plugins

    @classmethod
    def from_stage_string(cls, stages: str, **kwargs) -> "PipelineConfig":
        """
        Create config from a stage string.

        Examples:
            "1,2,3" -> stages 1, 2, 3
            "1-3" -> stages 1, 2, 3
            "1-5" -> all stages
        """
        enabled = set()
        for part in stages.split(","):
            part = part.strip()
            if "-" in part:
                start, end = part.split("-", 1)
                for i in range(int(start), int(end) + 1):
                    enabled.add(i)
            else:
                enabled.add(int(part))
        return cls(enabled_stages=enabled, **kwargs)

    @classmethod
    def default(cls) -> "PipelineConfig":
        """Create a default configuration with all stages enabled."""
        return cls()

    @classmethod
    def minimal(cls) -> "PipelineConfig":
        """Create a minimal configuration with only splitting and extraction."""
        return cls(enabled_stages={1, 2})


# Stage name mapping
STAGE_NAMES = {
    1: "splitting",
    2: "extraction",
    3: "qualification",
    4: "labeling",
    5: "taxonomy",
}


def get_stage_name(stage: int) -> str:
    """Get the human-readable name for a stage."""
    return STAGE_NAMES.get(stage, f"stage_{stage}")

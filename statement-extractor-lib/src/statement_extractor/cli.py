"""
Command-line interface for statement extraction.

Usage:
    corp-extractor "Your text here"
    corp-extractor -f input.txt
    corp-extractor pipeline "Your text here" --stages 1-5
    corp-extractor plugins list
"""

import json
import logging
import sys
from typing import Optional

import click


def _configure_logging(verbose: bool) -> None:
    """Configure logging for the extraction pipeline."""
    level = logging.DEBUG if verbose else logging.WARNING

    # Configure root logger for statement_extractor package
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
        force=True,
    )

    # Set level for all statement_extractor loggers
    for logger_name in [
        "statement_extractor",
        "statement_extractor.extractor",
        "statement_extractor.scoring",
        "statement_extractor.predicate_comparer",
        "statement_extractor.canonicalization",
        "statement_extractor.gliner_extraction",
        "statement_extractor.pipeline",
    ]:
        logging.getLogger(logger_name).setLevel(level)


from . import __version__
from .models import (
    ExtractionOptions,
    PredicateComparisonConfig,
    PredicateTaxonomy,
    ScoringConfig,
)


@click.group(invoke_without_command=True)
@click.pass_context
@click.argument("text", required=False)
@click.option("-f", "--file", "input_file", type=click.Path(exists=True), help="Read input from file")
@click.option(
    "-o", "--output",
    type=click.Choice(["table", "json", "xml"], case_sensitive=False),
    default="table",
    help="Output format (default: table)"
)
@click.option("--json", "output_json", is_flag=True, help="Output as JSON (shortcut for -o json)")
@click.option("--xml", "output_xml", is_flag=True, help="Output as XML (shortcut for -o xml)")
# Beam search options
@click.option("-b", "--beams", type=int, default=4, help="Number of beams for diverse beam search (default: 4)")
@click.option("--diversity", type=float, default=1.0, help="Diversity penalty for beam search (default: 1.0)")
@click.option("--max-tokens", type=int, default=2048, help="Maximum tokens to generate (default: 2048)")
# Deduplication options
@click.option("--no-dedup", is_flag=True, help="Disable deduplication")
@click.option("--no-embeddings", is_flag=True, help="Disable embedding-based deduplication (faster)")
@click.option("--no-merge", is_flag=True, help="Disable beam merging (select single best beam)")
@click.option("--no-gliner", is_flag=True, help="Disable GLiNER2 extraction (use raw model output)")
@click.option("--predicates", type=str, help="Comma-separated list of predicate types for GLiNER2 relation extraction")
@click.option("--all-triples", is_flag=True, help="Keep all candidate triples instead of selecting best per source")
@click.option("--dedup-threshold", type=float, default=0.65, help="Similarity threshold for deduplication (default: 0.65)")
# Quality options
@click.option("--min-confidence", type=float, default=0.0, help="Minimum confidence threshold 0-1 (default: 0)")
# Taxonomy options
@click.option("--taxonomy", type=click.Path(exists=True), help="Load predicate taxonomy from file (one per line)")
@click.option("--taxonomy-threshold", type=float, default=0.5, help="Similarity threshold for taxonomy matching (default: 0.5)")
# Device options
@click.option("--device", type=click.Choice(["auto", "cuda", "mps", "cpu"]), default="auto", help="Device to use (default: auto)")
# Output options
@click.option("-v", "--verbose", is_flag=True, help="Show verbose output with confidence scores")
@click.option("-q", "--quiet", is_flag=True, help="Suppress progress messages")
@click.version_option(version=__version__)
def main(
    ctx,
    text: Optional[str],
    input_file: Optional[str],
    output: str,
    output_json: bool,
    output_xml: bool,
    beams: int,
    diversity: float,
    max_tokens: int,
    no_dedup: bool,
    no_embeddings: bool,
    no_merge: bool,
    no_gliner: bool,
    predicates: Optional[str],
    all_triples: bool,
    dedup_threshold: float,
    min_confidence: float,
    taxonomy: Optional[str],
    taxonomy_threshold: float,
    device: str,
    verbose: bool,
    quiet: bool,
):
    """
    Extract structured statements from text.

    TEXT can be provided as an argument, read from a file with -f, or piped via stdin.

    \b
    Examples:
        corp-extractor "Apple announced a new iPhone."
        corp-extractor -f article.txt --json
        corp-extractor -f article.txt -o json --beams 8
        cat article.txt | corp-extractor -
        echo "Tim Cook is CEO of Apple." | corp-extractor - --verbose

    \b
    Subcommands:
        pipeline   Run the full 5-stage extraction pipeline
        plugins    List or inspect available plugins

    \b
    Output formats:
        table  Human-readable table (default)
        json   JSON with full metadata
        xml    Raw XML from model
    """
    # If a subcommand is being invoked, skip the main logic
    if ctx.invoked_subcommand is not None:
        return

    # Configure logging based on verbose flag
    _configure_logging(verbose)

    # Determine output format
    if output_json:
        output = "json"
    elif output_xml:
        output = "xml"

    # Get input text
    input_text = _get_input_text(text, input_file)
    if not input_text:
        # Show help if no input
        click.echo(ctx.get_help())
        return

    if not quiet:
        click.echo(f"Processing {len(input_text)} characters...", err=True)

    # Load taxonomy if provided
    predicate_taxonomy = None
    if taxonomy:
        predicate_taxonomy = PredicateTaxonomy.from_file(taxonomy)
        if not quiet:
            click.echo(f"Loaded taxonomy with {len(predicate_taxonomy.predicates)} predicates", err=True)

    # Configure predicate comparison
    predicate_config = PredicateComparisonConfig(
        similarity_threshold=taxonomy_threshold,
        dedup_threshold=dedup_threshold,
    )

    # Configure scoring
    scoring_config = ScoringConfig(min_confidence=min_confidence)

    # Parse predicates if provided
    predicate_list = None
    if predicates:
        predicate_list = [p.strip() for p in predicates.split(",") if p.strip()]
        if not quiet:
            click.echo(f"Using predicate list: {predicate_list}", err=True)

    # Configure extraction options
    options = ExtractionOptions(
        num_beams=beams,
        diversity_penalty=diversity,
        max_new_tokens=max_tokens,
        deduplicate=not no_dedup,
        embedding_dedup=not no_embeddings,
        merge_beams=not no_merge,
        use_gliner_extraction=not no_gliner,
        predicates=predicate_list,
        all_triples=all_triples,
        predicate_taxonomy=predicate_taxonomy,
        predicate_config=predicate_config,
        scoring_config=scoring_config,
        verbose=verbose,
    )

    # Import here to allow --help without loading torch
    from .extractor import StatementExtractor

    # Create extractor with specified device
    device_arg = None if device == "auto" else device
    extractor = StatementExtractor(device=device_arg)

    if not quiet:
        click.echo(f"Using device: {extractor.device}", err=True)

    # Run extraction
    try:
        if output == "xml":
            result = extractor.extract_as_xml(input_text, options)
            click.echo(result)
        elif output == "json":
            result = extractor.extract_as_json(input_text, options)
            click.echo(result)
        else:
            # Table format
            result = extractor.extract(input_text, options)
            _print_table(result, verbose)
    except Exception as e:
        logging.exception("Error extracting statements:")
        raise click.ClickException(f"Extraction failed: {e}")


# =============================================================================
# Pipeline command
# =============================================================================

@main.command("pipeline")
@click.argument("text", required=False)
@click.option("-f", "--file", "input_file", type=click.Path(exists=True), help="Read input from file")
@click.option(
    "--stages",
    type=str,
    default="1-5",
    help="Stages to run (e.g., '1,2,3' or '1-3' or '1-5')"
)
@click.option(
    "--skip-stages",
    type=str,
    default=None,
    help="Stages to skip (e.g., '4,5')"
)
@click.option(
    "--plugins",
    "enabled_plugins",
    type=str,
    default=None,
    help="Plugins to enable (comma-separated names)"
)
@click.option(
    "--disable-plugins",
    type=str,
    default=None,
    help="Plugins to disable (comma-separated names)"
)
@click.option(
    "-o", "--output",
    type=click.Choice(["table", "json", "yaml", "triples"], case_sensitive=False),
    default="table",
    help="Output format (default: table)"
)
@click.option("-v", "--verbose", is_flag=True, help="Show verbose output")
@click.option("-q", "--quiet", is_flag=True, help="Suppress progress messages")
def pipeline_cmd(
    text: Optional[str],
    input_file: Optional[str],
    stages: str,
    skip_stages: Optional[str],
    enabled_plugins: Optional[str],
    disable_plugins: Optional[str],
    output: str,
    verbose: bool,
    quiet: bool,
):
    """
    Run the full 5-stage extraction pipeline.

    \b
    Stages:
        1. Splitting      - Text → Raw triples (T5-Gemma)
        2. Extraction     - Raw triples → Typed statements (GLiNER2)
        3. Qualification  - Add qualifiers and identifiers
        4. Canonicalization - Resolve to canonical forms
        5. Labeling       - Apply sentiment, relation type, confidence

    \b
    Examples:
        corp-extractor pipeline "Apple CEO Tim Cook announced..."
        corp-extractor pipeline -f article.txt --stages 1-3
        corp-extractor pipeline "..." --plugins gleif,companies_house
        corp-extractor pipeline "..." --disable-plugins sec_edgar
    """
    _configure_logging(verbose)

    # Get input text
    input_text = _get_input_text(text, input_file)
    if not input_text:
        raise click.UsageError("No input provided. Provide text argument or use -f file.txt")

    if not quiet:
        click.echo(f"Processing {len(input_text)} characters through pipeline...", err=True)

    # Import pipeline components
    from .pipeline import ExtractionPipeline, PipelineConfig

    # Parse stages
    enabled_stages = _parse_stages(stages)
    if skip_stages:
        skip_set = _parse_stages(skip_stages)
        enabled_stages = enabled_stages - skip_set

    if not quiet:
        click.echo(f"Running stages: {sorted(enabled_stages)}", err=True)

    # Parse plugin selection
    enabled_plugin_set = None
    if enabled_plugins:
        enabled_plugin_set = {p.strip() for p in enabled_plugins.split(",") if p.strip()}

    disabled_plugin_set = set()
    if disable_plugins:
        disabled_plugin_set = {p.strip() for p in disable_plugins.split(",") if p.strip()}

    # Create config
    config = PipelineConfig(
        enabled_stages=enabled_stages,
        enabled_plugins=enabled_plugin_set,
        disabled_plugins=disabled_plugin_set,
    )

    # Run pipeline
    try:
        pipeline = ExtractionPipeline(config)
        ctx = pipeline.process(input_text)

        # Output results
        if output == "json":
            _print_pipeline_json(ctx)
        elif output == "yaml":
            _print_pipeline_yaml(ctx)
        elif output == "triples":
            _print_pipeline_triples(ctx)
        else:
            _print_pipeline_table(ctx, verbose)

        # Report errors/warnings
        if ctx.processing_errors and not quiet:
            click.echo(f"\nErrors: {len(ctx.processing_errors)}", err=True)
            for error in ctx.processing_errors:
                click.echo(f"  - {error}", err=True)

        if ctx.processing_warnings and verbose:
            click.echo(f"\nWarnings: {len(ctx.processing_warnings)}", err=True)
            for warning in ctx.processing_warnings:
                click.echo(f"  - {warning}", err=True)

    except Exception as e:
        logging.exception("Pipeline error:")
        raise click.ClickException(f"Pipeline failed: {e}")


def _parse_stages(stages_str: str) -> set[int]:
    """Parse stage string like '1,2,3' or '1-3' into a set of ints."""
    result = set()
    for part in stages_str.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            for i in range(int(start), int(end) + 1):
                result.add(i)
        else:
            result.add(int(part))
    return result


def _print_pipeline_json(ctx):
    """Print pipeline results as JSON."""
    output = {
        "statement_count": len(ctx.labeled_statements),
        "statements": [stmt.as_dict() for stmt in ctx.labeled_statements],
        "timings": ctx.stage_timings,
    }
    click.echo(json.dumps(output, indent=2))


def _print_pipeline_yaml(ctx):
    """Print pipeline results as YAML."""
    try:
        import yaml
        output = {
            "statement_count": len(ctx.labeled_statements),
            "statements": [stmt.as_dict() for stmt in ctx.labeled_statements],
            "timings": ctx.stage_timings,
        }
        click.echo(yaml.dump(output, default_flow_style=False))
    except ImportError:
        click.echo("YAML output requires PyYAML: pip install pyyaml", err=True)
        _print_pipeline_json(ctx)


def _print_pipeline_triples(ctx):
    """Print pipeline results as simple triples."""
    for stmt in ctx.labeled_statements:
        click.echo(f"{stmt.subject_fqn}\t{stmt.statement.predicate}\t{stmt.object_fqn}")


def _print_pipeline_table(ctx, verbose: bool):
    """Print pipeline results in table format."""
    if not ctx.labeled_statements:
        click.echo("No statements extracted.")
        return

    click.echo(f"\nExtracted {len(ctx.labeled_statements)} statement(s):\n")
    click.echo("-" * 80)

    for i, stmt in enumerate(ctx.labeled_statements, 1):
        # Subject FQN
        subj_fqn = stmt.subject_fqn
        obj_fqn = stmt.object_fqn

        click.echo(f"{i}. {subj_fqn}")
        click.echo(f"   --[{stmt.statement.predicate}]-->")
        click.echo(f"   {obj_fqn}")

        if verbose:
            # Show labels
            for label in stmt.labels:
                if isinstance(label.label_value, float):
                    click.echo(f"   {label.label_type}: {label.label_value:.3f}")
                else:
                    click.echo(f"   {label.label_type}: {label.label_value}")

            # Show source
            if stmt.statement.source_text:
                source = stmt.statement.source_text[:60] + "..." if len(stmt.statement.source_text) > 60 else stmt.statement.source_text
                click.echo(f"   Source: \"{source}\"")

        click.echo("-" * 80)

    # Show timings in verbose mode
    if verbose and ctx.stage_timings:
        click.echo("\nStage timings:")
        for stage, duration in ctx.stage_timings.items():
            click.echo(f"  {stage}: {duration:.3f}s")


# =============================================================================
# Plugins command
# =============================================================================

@main.command("plugins")
@click.argument("action", type=click.Choice(["list", "info"]))
@click.argument("plugin_name", required=False)
@click.option("--stage", type=int, help="Filter by stage number (1-5)")
def plugins_cmd(action: str, plugin_name: Optional[str], stage: Optional[int]):
    """
    List or inspect available plugins.

    \b
    Actions:
        list   List all available plugins
        info   Show details about a specific plugin

    \b
    Examples:
        corp-extractor plugins list
        corp-extractor plugins list --stage 3
        corp-extractor plugins info gleif_qualifier
    """
    # Import and load plugins
    _load_all_plugins()

    from .pipeline.registry import PluginRegistry

    if action == "list":
        plugins = PluginRegistry.list_plugins(stage=stage)
        if not plugins:
            click.echo("No plugins registered.")
            return

        # Group by stage
        by_stage: dict[int, list] = {}
        for plugin in plugins:
            stage_num = plugin["stage"]
            if stage_num not in by_stage:
                by_stage[stage_num] = []
            by_stage[stage_num].append(plugin)

        for stage_num in sorted(by_stage.keys()):
            stage_plugins = by_stage[stage_num]
            stage_name = stage_plugins[0]["stage_name"]
            click.echo(f"\nStage {stage_num}: {stage_name.title()}")
            click.echo("-" * 40)

            for p in stage_plugins:
                entity_types = p.get("entity_types", [])
                types_str = f" ({', '.join(entity_types)})" if entity_types else ""
                click.echo(f"  {p['name']}{types_str}  [priority: {p['priority']}]")

    elif action == "info":
        if not plugin_name:
            raise click.UsageError("Plugin name required for 'info' action")

        plugin = PluginRegistry.get_plugin(plugin_name)
        if not plugin:
            raise click.ClickException(f"Plugin not found: {plugin_name}")

        click.echo(f"\nPlugin: {plugin.name}")
        click.echo(f"Priority: {plugin.priority}")
        click.echo(f"Capabilities: {plugin.capabilities.name if plugin.capabilities else 'NONE'}")

        if plugin.description:
            click.echo(f"Description: {plugin.description}")

        if hasattr(plugin, "supported_entity_types"):
            types = [t.value for t in plugin.supported_entity_types]
            click.echo(f"Entity types: {', '.join(types)}")

        if hasattr(plugin, "label_type"):
            click.echo(f"Label type: {plugin.label_type}")

        if hasattr(plugin, "supported_identifier_types"):
            ids = plugin.supported_identifier_types
            if ids:
                click.echo(f"Supported identifiers: {', '.join(ids)}")

        if hasattr(plugin, "provided_identifier_types"):
            ids = plugin.provided_identifier_types
            if ids:
                click.echo(f"Provided identifiers: {', '.join(ids)}")


def _load_all_plugins():
    """Load all plugins by importing their modules."""
    # Import all plugin modules to trigger registration
    try:
        from .plugins import splitters, extractors, qualifiers, canonicalizers, labelers
        # The @PluginRegistry decorators will register plugins on import
    except ImportError as e:
        logging.debug(f"Some plugins failed to load: {e}")


# =============================================================================
# Helper functions
# =============================================================================

def _get_input_text(text: Optional[str], input_file: Optional[str]) -> Optional[str]:
    """Get input text from argument, file, or stdin."""
    if text == "-" or (text is None and input_file is None and not sys.stdin.isatty()):
        # Read from stdin
        return sys.stdin.read().strip()
    elif input_file:
        # Read from file
        with open(input_file, "r", encoding="utf-8") as f:
            return f.read().strip()
    elif text:
        return text.strip()
    return None


def _print_table(result, verbose: bool):
    """Print statements in a human-readable table format."""
    if not result.statements:
        click.echo("No statements extracted.")
        return

    click.echo(f"\nExtracted {len(result.statements)} statement(s):\n")
    click.echo("-" * 80)

    for i, stmt in enumerate(result.statements, 1):
        subject_type = f" ({stmt.subject.type.value})" if stmt.subject.type.value != "UNKNOWN" else ""
        object_type = f" ({stmt.object.type.value})" if stmt.object.type.value != "UNKNOWN" else ""

        click.echo(f"{i}. {stmt.subject.text}{subject_type}")
        click.echo(f"   --[{stmt.predicate}]-->")
        click.echo(f"   {stmt.object.text}{object_type}")

        if verbose:
            # Always show extraction method
            click.echo(f"   Method: {stmt.extraction_method.value}")

            if stmt.confidence_score is not None:
                click.echo(f"   Confidence: {stmt.confidence_score:.2f}")

            if stmt.canonical_predicate:
                click.echo(f"   Canonical: {stmt.canonical_predicate}")

            if stmt.was_reversed:
                click.echo(f"   (subject/object were swapped)")

            if stmt.source_text:
                source = stmt.source_text[:60] + "..." if len(stmt.source_text) > 60 else stmt.source_text
                click.echo(f"   Source: \"{source}\"")

        click.echo("-" * 80)


if __name__ == "__main__":
    main()

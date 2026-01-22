"""
Command-line interface for statement extraction.

Usage:
    corp-extractor split "Your text here"
    corp-extractor split -f input.txt
    corp-extractor pipeline "Your text here" --stages 1-5
    corp-extractor plugins list
"""

import json
import logging
import sys
from pathlib import Path
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
        "statement_extractor.plugins",
        "statement_extractor.plugins.extractors.gliner2",
        "statement_extractor.plugins.splitters",
        "statement_extractor.plugins.labelers",
        "statement_extractor.plugins.scrapers",
        "statement_extractor.plugins.scrapers.http",
        "statement_extractor.plugins.pdf",
        "statement_extractor.plugins.pdf.pypdf",
        "statement_extractor.document",
        "statement_extractor.document.loader",
        "statement_extractor.document.html_extractor",
        "statement_extractor.document.pipeline",
        "statement_extractor.document.chunker",
    ]:
        logging.getLogger(logger_name).setLevel(level)

    # Suppress noisy third-party loggers
    for noisy_logger in [
        "httpcore",
        "httpcore.http11",
        "httpcore.connection",
        "httpx",
        "urllib3",
        "huggingface_hub",
        "asyncio",
    ]:
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)


from . import __version__
from .models import (
    ExtractionOptions,
    PredicateComparisonConfig,
    PredicateTaxonomy,
    ScoringConfig,
)


@click.group()
@click.version_option(version=__version__)
def main():
    """
    Extract structured statements from text.

    \b
    Commands:
        split      Extract sub-statements from text (simple, fast)
        pipeline   Run the full 6-stage extraction pipeline
        document   Process documents with chunking and citations
        plugins    List or inspect available plugins
        db         Manage entity/organization embedding database

    \b
    Examples:
        corp-extractor split "Apple announced a new iPhone."
        corp-extractor split -f article.txt --json
        corp-extractor pipeline "Apple CEO Tim Cook announced..." --stages 1-3
        corp-extractor document process report.txt --title "Annual Report"
        corp-extractor plugins list
    """
    pass


# =============================================================================
# Split command (simple extraction)
# =============================================================================

@main.command("split")
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
def split_cmd(
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
    Extract sub-statements from text using T5-Gemma model.

    This command splits text into structured subject-predicate-object triples.
    It's fast and simple - use 'pipeline' for full entity resolution.

    \b
    Examples:
        corp-extractor split "Apple announced a new iPhone."
        corp-extractor split -f article.txt --json
        corp-extractor split -f article.txt -o json --beams 8
        cat article.txt | corp-extractor split -
        echo "Tim Cook is CEO of Apple." | corp-extractor split - --verbose

    \b
    Output formats:
        table  Human-readable table (default)
        json   JSON with full metadata
        xml    Raw XML from model
    """
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
        raise click.UsageError("No input provided. Provide text argument or use -f file.txt")

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
    default="1-6",
    help="Stages to run (e.g., '1,2,3' or '1-3' or '1-6')"
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
    "--no-default-predicates",
    is_flag=True,
    help="Disable default predicate taxonomy (GLiNER2 will only use entity extraction)"
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
    no_default_predicates: bool,
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

    # Import pipeline components (also loads plugins)
    from .pipeline import ExtractionPipeline, PipelineConfig
    _load_all_plugins()

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

    disabled_plugin_set = None
    if disable_plugins:
        disabled_plugin_set = {p.strip() for p in disable_plugins.split(",") if p.strip()}

    # Build extractor options
    extractor_options = {}
    if no_default_predicates:
        extractor_options["use_default_predicates"] = False
        if not quiet:
            click.echo("Default predicates disabled - using entity extraction only", err=True)

    # Create config - only pass disabled_plugins if user explicitly specified, otherwise use defaults
    config_kwargs: dict = {
        "enabled_stages": enabled_stages,
        "enabled_plugins": enabled_plugin_set,
        "extractor_options": extractor_options,
    }
    if disabled_plugin_set is not None:
        config_kwargs["disabled_plugins"] = disabled_plugin_set
    config = PipelineConfig(**config_kwargs)

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
        "statement_count": ctx.statement_count,
        "raw_triples": [t.model_dump() for t in ctx.raw_triples],
        "statements": [s.model_dump() for s in ctx.statements],
        "labeled_statements": [stmt.as_dict() for stmt in ctx.labeled_statements],
        "timings": ctx.stage_timings,
        "warnings": ctx.processing_warnings,
        "errors": ctx.processing_errors,
    }
    click.echo(json.dumps(output, indent=2, default=str))


def _print_pipeline_yaml(ctx):
    """Print pipeline results as YAML."""
    try:
        import yaml
        output = {
            "statement_count": ctx.statement_count,
            "statements": [stmt.as_dict() for stmt in ctx.labeled_statements],
            "timings": ctx.stage_timings,
        }
        click.echo(yaml.dump(output, default_flow_style=False))
    except ImportError:
        click.echo("YAML output requires PyYAML: pip install pyyaml", err=True)
        _print_pipeline_json(ctx)


def _print_pipeline_triples(ctx):
    """Print pipeline results as simple triples."""
    if ctx.labeled_statements:
        for stmt in ctx.labeled_statements:
            click.echo(f"{stmt.subject_fqn}\t{stmt.statement.predicate}\t{stmt.object_fqn}")
    elif ctx.statements:
        for stmt in ctx.statements:
            click.echo(f"{stmt.subject.text}\t{stmt.predicate}\t{stmt.object.text}")
    elif ctx.raw_triples:
        for triple in ctx.raw_triples:
            click.echo(f"{triple.subject_text}\t{triple.predicate_text}\t{triple.object_text}")


def _print_pipeline_table(ctx, verbose: bool):
    """Print pipeline results in table format."""
    # Try labeled statements first, then statements, then raw triples
    if ctx.labeled_statements:
        click.echo(f"\nExtracted {len(ctx.labeled_statements)} statement(s):\n")
        click.echo("-" * 80)

        for i, stmt in enumerate(ctx.labeled_statements, 1):
            click.echo(f"{i}. {stmt.subject_fqn}")
            click.echo(f"   --[{stmt.statement.predicate}]-->")
            click.echo(f"   {stmt.object_fqn}")

            # Show labels (always in recent versions, not just verbose)
            for label in stmt.labels:
                if isinstance(label.label_value, float):
                    click.echo(f"   {label.label_type}: {label.label_value:.3f}")
                else:
                    click.echo(f"   {label.label_type}: {label.label_value}")

            # Show top taxonomy results (sorted by confidence)
            if stmt.taxonomy_results:
                sorted_taxonomy = sorted(stmt.taxonomy_results, key=lambda t: t.confidence, reverse=True)
                top_taxonomy = sorted_taxonomy[:5]  # Show top 5
                taxonomy_strs = [f"{t.category}:{t.label} ({t.confidence:.2f})" for t in top_taxonomy]
                click.echo(f"   topics: {', '.join(taxonomy_strs)}")
                if len(sorted_taxonomy) > 5:
                    click.echo(f"   ... and {len(sorted_taxonomy) - 5} more topics")

            if verbose and stmt.statement.source_text:
                source = stmt.statement.source_text[:60] + "..." if len(stmt.statement.source_text) > 60 else stmt.statement.source_text
                click.echo(f"   Source: \"{source}\"")

            click.echo("-" * 80)

    elif ctx.statements:
        click.echo(f"\nExtracted {len(ctx.statements)} statement(s):\n")
        click.echo("-" * 80)

        for i, stmt in enumerate(ctx.statements, 1):
            subj_type = f" ({stmt.subject.type.value})" if stmt.subject.type.value != "UNKNOWN" else ""
            obj_type = f" ({stmt.object.type.value})" if stmt.object.type.value != "UNKNOWN" else ""

            click.echo(f"{i}. {stmt.subject.text}{subj_type}")
            click.echo(f"   --[{stmt.predicate}]-->")
            click.echo(f"   {stmt.object.text}{obj_type}")

            if verbose and stmt.confidence_score is not None:
                click.echo(f"   Confidence: {stmt.confidence_score:.2f}")

            click.echo("-" * 80)

    elif ctx.raw_triples:
        click.echo(f"\nExtracted {len(ctx.raw_triples)} raw triple(s):\n")
        click.echo("-" * 80)

        for i, triple in enumerate(ctx.raw_triples, 1):
            click.echo(f"{i}. {triple.subject_text}")
            click.echo(f"   --[{triple.predicate_text}]-->")
            click.echo(f"   {triple.object_text}")

            if verbose:
                click.echo(f"   Confidence: {triple.confidence:.2f}")
                if triple.source_sentence:
                    source = triple.source_sentence[:60] + "..." if len(triple.source_sentence) > 60 else triple.source_sentence
                    click.echo(f"   Source: \"{source}\"")

            click.echo("-" * 80)

    else:
        click.echo("No statements extracted.")
        return

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
        from .plugins import splitters, extractors, qualifiers, labelers, taxonomy
        # The @PluginRegistry decorators will register plugins on import
        _ = splitters, extractors, qualifiers, labelers, taxonomy  # Silence unused warnings
    except ImportError as e:
        logging.debug(f"Some plugins failed to load: {e}")


# =============================================================================
# Database commands
# =============================================================================

@main.group("db")
def db_cmd():
    """
    Manage entity/organization embedding database.

    \b
    Commands:
        import-gleif           Import GLEIF LEI data (~3M records)
        import-sec             Import SEC Edgar bulk data (~100K+ filers)
        import-companies-house Import UK Companies House (~5M records)
        import-wikidata        Import Wikidata organizations
        status                 Show database status
        search                 Search for an organization
        download               Download database from HuggingFace
        upload                 Upload database with lite/compressed variants
        create-lite            Create lite version (no record data)
        compress               Compress database with gzip

    \b
    Examples:
        corp-extractor db import-sec --download
        corp-extractor db import-gleif --download --limit 100000
        corp-extractor db status
        corp-extractor db search "Apple Inc"
        corp-extractor db upload entities.db
    """
    pass


@db_cmd.command("gleif-info")
def db_gleif_info():
    """
    Show information about the latest available GLEIF data file.

    \b
    Examples:
        corp-extractor db gleif-info
    """
    from .database.importers import GleifImporter

    importer = GleifImporter()

    try:
        info = importer.get_latest_file_info()
        record_count = info.get('record_count')

        click.echo("\nLatest GLEIF Data File")
        click.echo("=" * 40)
        click.echo(f"File ID: {info['id']}")
        click.echo(f"Publish Date: {info['publish_date']}")
        click.echo(f"Record Count: {record_count:,}" if record_count else "Record Count: unknown")

        delta = info.get("delta_from_last_file", {})
        if delta:
            click.echo(f"\nChanges from previous file:")
            if delta.get('new'):
                click.echo(f"  New: {delta.get('new'):,}")
            if delta.get('updated'):
                click.echo(f"  Updated: {delta.get('updated'):,}")
            if delta.get('retired'):
                click.echo(f"  Retired: {delta.get('retired'):,}")

    except Exception as e:
        raise click.ClickException(f"Failed to get GLEIF info: {e}")


@db_cmd.command("import-gleif")
@click.argument("file_path", type=click.Path(exists=True), required=False)
@click.option("--download", is_flag=True, help="Download latest GLEIF file before importing")
@click.option("--force", is_flag=True, help="Force re-download even if cached")
@click.option("--db", "db_path", type=click.Path(), help="Database path (default: ~/.cache/corp-extractor/entities.db)")
@click.option("--limit", type=int, help="Limit number of records to import")
@click.option("--batch-size", type=int, default=50000, help="Batch size for commits (default: 50000)")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
def db_import_gleif(file_path: Optional[str], download: bool, force: bool, db_path: Optional[str], limit: Optional[int], batch_size: int, verbose: bool):
    """
    Import GLEIF LEI data into the entity database.

    If no file path is provided and --download is set, downloads the latest
    GLEIF data file automatically. Downloaded files are cached and reused
    unless --force is specified.

    \b
    Examples:
        corp-extractor db import-gleif /path/to/lei-records.xml
        corp-extractor db import-gleif --download
        corp-extractor db import-gleif --download --limit 10000
        corp-extractor db import-gleif --download --force  # Re-download
    """
    _configure_logging(verbose)

    from .database import OrganizationDatabase, CompanyEmbedder
    from .database.importers import GleifImporter

    importer = GleifImporter()

    # Handle file path
    if file_path is None:
        if not download:
            raise click.UsageError("Either provide a file path or use --download to fetch the latest GLEIF data")
        click.echo("Downloading latest GLEIF data...", err=True)
        file_path = str(importer.download_latest(force=force))
    elif download:
        click.echo("Downloading latest GLEIF data (ignoring provided file path)...", err=True)
        file_path = str(importer.download_latest(force=force))

    click.echo(f"Importing GLEIF data from {file_path}...", err=True)

    # Initialize components
    embedder = CompanyEmbedder()
    database = OrganizationDatabase(db_path=db_path, embedding_dim=embedder.embedding_dim)

    # Import records in batches
    records = []
    count = 0

    for record in importer.import_from_file(file_path, limit=limit):
        records.append(record)

        if len(records) >= batch_size:
            # Embed and insert batch
            names = [r.name for r in records]
            embeddings = embedder.embed_batch(names)
            database.insert_batch(records, embeddings)
            count += len(records)
            click.echo(f"Imported {count} records...", err=True)
            records = []

    # Final batch
    if records:
        names = [r.name for r in records]
        embeddings = embedder.embed_batch(names)
        database.insert_batch(records, embeddings)
        count += len(records)

    click.echo(f"\nImported {count} GLEIF records successfully.", err=True)
    database.close()


@db_cmd.command("import-sec")
@click.option("--download", is_flag=True, help="Download bulk submissions.zip (~500MB, ~100K+ filers)")
@click.option("--file", "file_path", type=click.Path(exists=True), help="Local file (submissions.zip or company_tickers.json)")
@click.option("--db", "db_path", type=click.Path(), help="Database path")
@click.option("--limit", type=int, help="Limit number of records")
@click.option("--batch-size", type=int, default=10000, help="Batch size (default: 10000)")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
def db_import_sec(download: bool, file_path: Optional[str], db_path: Optional[str], limit: Optional[int], batch_size: int, verbose: bool):
    """
    Import SEC Edgar data into the entity database.

    By default, downloads the bulk submissions.zip file which contains
    ALL SEC filers (~100K+), not just companies with ticker symbols (~10K).

    \b
    Examples:
        corp-extractor db import-sec --download
        corp-extractor db import-sec --download --limit 50000
        corp-extractor db import-sec --file /path/to/submissions.zip
        corp-extractor db import-sec --file /path/to/company_tickers.json  # legacy
    """
    _configure_logging(verbose)

    from .database import OrganizationDatabase, CompanyEmbedder
    from .database.importers import SecEdgarImporter

    if not download and not file_path:
        raise click.UsageError("Either --download or --file is required")

    # Initialize components
    embedder = CompanyEmbedder()
    database = OrganizationDatabase(db_path=db_path, embedding_dim=embedder.embedding_dim)
    importer = SecEdgarImporter()

    # Get records
    if file_path:
        click.echo(f"Importing SEC Edgar data from {file_path}...", err=True)
        record_iter = importer.import_from_file(file_path, limit=limit)
    else:
        click.echo("Downloading SEC submissions.zip (~500MB)...", err=True)
        record_iter = importer.import_from_url(limit=limit)

    # Import records in batches
    records = []
    count = 0

    for record in record_iter:
        records.append(record)

        if len(records) >= batch_size:
            names = [r.name for r in records]
            embeddings = embedder.embed_batch(names)
            database.insert_batch(records, embeddings)
            count += len(records)
            click.echo(f"Imported {count} records...", err=True)
            records = []

    # Final batch
    if records:
        names = [r.name for r in records]
        embeddings = embedder.embed_batch(names)
        database.insert_batch(records, embeddings)
        count += len(records)

    click.echo(f"\nImported {count} SEC Edgar records successfully.", err=True)
    database.close()


@db_cmd.command("import-wikidata")
@click.option("--db", "db_path", type=click.Path(), help="Database path")
@click.option("--limit", type=int, help="Limit number of records")
@click.option("--batch-size", type=int, default=1000, help="Batch size for commits (default: 1000)")
@click.option("--type", "query_type", type=click.Choice(["lei", "ticker", "public", "business", "organization", "nonprofit", "government"]), default="lei",
              help="Query type to use for fetching data")
@click.option("--all", "import_all", is_flag=True, help="Run all query types sequentially")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
def db_import_wikidata(db_path: Optional[str], limit: Optional[int], batch_size: int, query_type: str, import_all: bool, verbose: bool):
    """
    Import organization data from Wikidata via SPARQL.

    Uses simplified SPARQL queries that avoid timeouts on Wikidata's endpoint.
    Query types target different organization categories.

    \b
    Query types:
        lei          Companies with LEI codes (fastest, most reliable)
        ticker       Companies listed on stock exchanges
        public       Direct instances of "public company" (Q891723)
        business     Direct instances of "business enterprise" (Q4830453)
        organization All organizations (Q43229) - NGOs, associations, etc.
        nonprofit    Non-profit organizations (Q163740)
        government   Government agencies (Q327333)

    \b
    Examples:
        corp-extractor db import-wikidata --limit 10
        corp-extractor db import-wikidata --type organization --limit 1000
        corp-extractor db import-wikidata --type nonprofit --limit 5000
        corp-extractor db import-wikidata --all --limit 10000
    """
    _configure_logging(verbose)

    from .database import OrganizationDatabase, CompanyEmbedder
    from .database.importers import WikidataImporter

    click.echo(f"Importing Wikidata organization data via SPARQL (type={query_type}, all={import_all})...", err=True)

    # Initialize components
    embedder = CompanyEmbedder()
    database = OrganizationDatabase(db_path=db_path, embedding_dim=embedder.embedding_dim)
    importer = WikidataImporter(batch_size=500)  # Smaller SPARQL batch size for reliability

    # Import records in batches
    records = []
    count = 0

    for record in importer.import_from_sparql(limit=limit, query_type=query_type, import_all=import_all):
        records.append(record)

        if len(records) >= batch_size:
            names = [r.name for r in records]
            embeddings = embedder.embed_batch(names)
            database.insert_batch(records, embeddings)
            count += len(records)
            click.echo(f"Imported {count} records...", err=True)
            records = []

    # Final batch
    if records:
        names = [r.name for r in records]
        embeddings = embedder.embed_batch(names)
        database.insert_batch(records, embeddings)
        count += len(records)

    click.echo(f"\nImported {count} Wikidata records successfully.", err=True)
    database.close()


@db_cmd.command("import-companies-house")
@click.option("--download", is_flag=True, help="Download bulk data file (free, no API key needed)")
@click.option("--force", is_flag=True, help="Force re-download even if cached")
@click.option("--file", "file_path", type=click.Path(exists=True), help="Local Companies House CSV/JSON file")
@click.option("--search", "search_terms", type=str, help="Comma-separated search terms (requires API key)")
@click.option("--db", "db_path", type=click.Path(), help="Database path")
@click.option("--limit", type=int, help="Limit number of records")
@click.option("--batch-size", type=int, default=50000, help="Batch size for commits (default: 50000)")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
def db_import_companies_house(
    download: bool,
    force: bool,
    file_path: Optional[str],
    search_terms: Optional[str],
    db_path: Optional[str],
    limit: Optional[int],
    batch_size: int,
    verbose: bool,
):
    """
    Import UK Companies House data into the entity database.

    \b
    Options:
    --download    Download free bulk data (all UK companies, ~5M records)
    --file        Import from local CSV/JSON file
    --search      Search via API (requires COMPANIES_HOUSE_API_KEY)

    \b
    Examples:
        corp-extractor db import-companies-house --download
        corp-extractor db import-companies-house --download --limit 100000
        corp-extractor db import-companies-house --file /path/to/companies.csv
        corp-extractor db import-companies-house --search "bank,insurance"
    """
    _configure_logging(verbose)

    from .database import OrganizationDatabase, CompanyEmbedder
    from .database.importers import CompaniesHouseImporter

    if not file_path and not search_terms and not download:
        raise click.UsageError("Either --download, --file, or --search is required")

    click.echo("Importing Companies House data...", err=True)

    # Initialize components
    embedder = CompanyEmbedder()
    database = OrganizationDatabase(db_path=db_path, embedding_dim=embedder.embedding_dim)
    importer = CompaniesHouseImporter()

    # Get records
    if download:
        # Download bulk data file
        csv_path = importer.download_bulk_data(force=force)
        click.echo(f"Using bulk data file: {csv_path}", err=True)
        record_iter = importer.import_from_file(csv_path, limit=limit)
    elif file_path:
        record_iter = importer.import_from_file(file_path, limit=limit)
    else:
        terms = [t.strip() for t in search_terms.split(",") if t.strip()]
        click.echo(f"Searching for: {terms}", err=True)
        record_iter = importer.import_from_search(
            search_terms=terms,
            limit_per_term=limit or 100,
            total_limit=limit,
        )

    # Import records in batches
    records = []
    count = 0

    for record in record_iter:
        records.append(record)

        if len(records) >= batch_size:
            names = [r.name for r in records]
            embeddings = embedder.embed_batch(names)
            database.insert_batch(records, embeddings)
            count += len(records)
            click.echo(f"Imported {count} records...", err=True)
            records = []

    # Final batch
    if records:
        names = [r.name for r in records]
        embeddings = embedder.embed_batch(names)
        database.insert_batch(records, embeddings)
        count += len(records)

    click.echo(f"\nImported {count} Companies House records successfully.", err=True)
    database.close()


@db_cmd.command("status")
@click.option("--db", "db_path", type=click.Path(), help="Database path")
def db_status(db_path: Optional[str]):
    """
    Show database status and statistics.

    \b
    Examples:
        corp-extractor db status
        corp-extractor db status --db /path/to/entities.db
    """
    from .database import OrganizationDatabase

    try:
        database = OrganizationDatabase(db_path=db_path)
        stats = database.get_stats()

        click.echo("\nEntity Database Status")
        click.echo("=" * 40)
        click.echo(f"Total records: {stats.total_records:,}")
        click.echo(f"Embedding dimension: {stats.embedding_dimension}")
        click.echo(f"Database size: {stats.database_size_bytes / 1024 / 1024:.2f} MB")

        # Check for missing embeddings
        missing_embeddings = database.get_missing_embedding_count()
        if missing_embeddings > 0:
            click.echo(f"\n⚠️  Missing embeddings: {missing_embeddings:,}")
            click.echo("   Run 'corp-extractor db repair-embeddings' to fix")

        if stats.by_source:
            click.echo("\nRecords by source:")
            for source, count in stats.by_source.items():
                click.echo(f"  {source}: {count:,}")

        database.close()

    except Exception as e:
        raise click.ClickException(f"Failed to read database: {e}")


@db_cmd.command("search")
@click.argument("query")
@click.option("--db", "db_path", type=click.Path(), help="Database path")
@click.option("--top-k", type=int, default=10, help="Number of results")
@click.option("--source", type=click.Choice(["gleif", "sec_edgar", "companies_house", "wikipedia"]), help="Filter by source")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
def db_search(query: str, db_path: Optional[str], top_k: int, source: Optional[str], verbose: bool):
    """
    Search for an organization in the database.

    \b
    Examples:
        corp-extractor db search "Apple Inc"
        corp-extractor db search "Microsoft" --source sec_edgar
    """
    _configure_logging(verbose)

    from .database import OrganizationDatabase, CompanyEmbedder

    embedder = CompanyEmbedder()
    database = OrganizationDatabase(db_path=db_path)

    click.echo(f"Searching for: {query}", err=True)

    # Embed query
    query_embedding = embedder.embed(query)

    # Search
    results = database.search(query_embedding, top_k=top_k, source_filter=source)

    if not results:
        click.echo("No results found.")
        return

    click.echo(f"\nTop {len(results)} matches:")
    click.echo("-" * 60)

    for i, (record, similarity) in enumerate(results, 1):
        click.echo(f"{i}. {record.legal_name}")
        click.echo(f"   Source: {record.source} | ID: {record.source_id}")
        click.echo(f"   Canonical ID: {record.canonical_id}")
        click.echo(f"   Similarity: {similarity:.4f}")
        if verbose and record.record:
            if record.record.get("ticker"):
                click.echo(f"   Ticker: {record.record['ticker']}")
            if record.record.get("jurisdiction"):
                click.echo(f"   Jurisdiction: {record.record['jurisdiction']}")
        click.echo()

    database.close()


@db_cmd.command("download")
@click.option("--repo", type=str, default="Corp-o-Rate-Community/entity-references", help="HuggingFace repo ID")
@click.option("--db", "db_path", type=click.Path(), help="Output path for database")
@click.option("--full", is_flag=True, help="Download full version (larger, includes record metadata)")
@click.option("--no-compress", is_flag=True, help="Download uncompressed version (slower)")
@click.option("--force", is_flag=True, help="Force re-download")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
def db_download(repo: str, db_path: Optional[str], full: bool, no_compress: bool, force: bool, verbose: bool):
    """
    Download entity database from HuggingFace Hub.

    By default downloads the lite version (smaller, without record metadata).
    Use --full for the complete database with all source record data.

    \b
    Examples:
        corp-extractor db download
        corp-extractor db download --full
        corp-extractor db download --repo my-org/my-entity-db
    """
    _configure_logging(verbose)
    from .database.hub import download_database

    filename = "entities.db" if full else "entities-lite.db"
    click.echo(f"Downloading {'full ' if full else 'lite '}database from {repo}...", err=True)

    try:
        path = download_database(
            repo_id=repo,
            filename=filename,
            force_download=force,
            prefer_compressed=not no_compress,
        )
        click.echo(f"Database downloaded to: {path}")
    except Exception as e:
        raise click.ClickException(f"Download failed: {e}")


@db_cmd.command("upload")
@click.argument("db_path", type=click.Path(exists=True), required=False)
@click.option("--repo", type=str, default="Corp-o-Rate-Community/entity-references", help="HuggingFace repo ID")
@click.option("--message", type=str, default="Update entity database", help="Commit message")
@click.option("--no-lite", is_flag=True, help="Skip creating lite version (without record data)")
@click.option("--no-compress", is_flag=True, help="Skip creating compressed versions")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
def db_upload(db_path: Optional[str], repo: str, message: str, no_lite: bool, no_compress: bool, verbose: bool):
    """
    Upload entity database to HuggingFace Hub with variants.

    If no path is provided, uploads from the default cache location.

    By default uploads:
    - entities.db (full database)
    - entities-lite.db (without record data, smaller)
    - entities.db.gz (compressed full)
    - entities-lite.db.gz (compressed lite)

    Requires HF_TOKEN environment variable to be set.

    \b
    Examples:
        corp-extractor db upload
        corp-extractor db upload /path/to/entities.db
        corp-extractor db upload --no-lite --no-compress
        corp-extractor db upload --repo my-org/my-entity-db
    """
    _configure_logging(verbose)
    from .database.hub import upload_database_with_variants, DEFAULT_CACHE_DIR, DEFAULT_DB_FULL_FILENAME

    # Use default cache location if no path provided
    if db_path is None:
        db_path = str(DEFAULT_CACHE_DIR / DEFAULT_DB_FULL_FILENAME)
        if not Path(db_path).exists():
            raise click.ClickException(
                f"Database not found at default location: {db_path}\n"
                "Build the database first with import commands, or specify a path."
            )

    click.echo(f"Uploading {db_path} to {repo}...", err=True)
    if not no_lite:
        click.echo("  - Creating lite version (without record data)", err=True)
    if not no_compress:
        click.echo("  - Creating compressed versions", err=True)

    try:
        results = upload_database_with_variants(
            db_path=db_path,
            repo_id=repo,
            commit_message=message,
            include_lite=not no_lite,
            include_compressed=not no_compress,
        )
        click.echo(f"\nUploaded {len(results)} file(s) successfully:")
        for filename, url in results.items():
            click.echo(f"  - {filename}")
    except Exception as e:
        raise click.ClickException(f"Upload failed: {e}")


@db_cmd.command("create-lite")
@click.argument("db_path", type=click.Path(exists=True))
@click.option("-o", "--output", type=click.Path(), help="Output path (default: adds -lite suffix)")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
def db_create_lite(db_path: str, output: Optional[str], verbose: bool):
    """
    Create a lite version of the database without record data.

    The lite version strips the `record` column (full source data),
    keeping only core fields and embeddings. This significantly
    reduces file size while maintaining search functionality.

    \b
    Examples:
        corp-extractor db create-lite entities.db
        corp-extractor db create-lite entities.db -o entities-lite.db
    """
    _configure_logging(verbose)
    from .database.hub import create_lite_database

    click.echo(f"Creating lite database from {db_path}...", err=True)

    try:
        lite_path = create_lite_database(db_path, output)
        click.echo(f"Lite database created: {lite_path}")
    except Exception as e:
        raise click.ClickException(f"Failed to create lite database: {e}")


@db_cmd.command("compress")
@click.argument("db_path", type=click.Path(exists=True))
@click.option("-o", "--output", type=click.Path(), help="Output path (default: adds .gz suffix)")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
def db_compress(db_path: str, output: Optional[str], verbose: bool):
    """
    Compress a database file using gzip.

    \b
    Examples:
        corp-extractor db compress entities.db
        corp-extractor db compress entities.db -o entities.db.gz
    """
    _configure_logging(verbose)
    from .database.hub import compress_database

    click.echo(f"Compressing {db_path}...", err=True)

    try:
        compressed_path = compress_database(db_path, output)
        click.echo(f"Compressed database created: {compressed_path}")
    except Exception as e:
        raise click.ClickException(f"Compression failed: {e}")


@db_cmd.command("repair-embeddings")
@click.option("--db", "db_path", type=click.Path(), help="Database path")
@click.option("--batch-size", type=int, default=1000, help="Batch size for embedding generation (default: 1000)")
@click.option("--source", type=str, help="Only repair specific source (gleif, sec_edgar, etc.)")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
def db_repair_embeddings(db_path: Optional[str], batch_size: int, source: Optional[str], verbose: bool):
    """
    Generate missing embeddings for organizations in the database.

    This repairs databases where organizations were imported without embeddings
    being properly stored in the organization_embeddings table.

    \b
    Examples:
        corp-extractor db repair-embeddings
        corp-extractor db repair-embeddings --source wikipedia
        corp-extractor db repair-embeddings --batch-size 500
    """
    _configure_logging(verbose)

    from .database import OrganizationDatabase, CompanyEmbedder

    database = OrganizationDatabase(db_path=db_path)
    embedder = CompanyEmbedder()

    # Check how many need repair
    missing_count = database.get_missing_embedding_count()
    if missing_count == 0:
        click.echo("All organizations have embeddings. Nothing to repair.")
        database.close()
        return

    click.echo(f"Found {missing_count:,} organizations without embeddings.", err=True)
    click.echo("Generating embeddings...", err=True)

    # Process in batches
    org_ids = []
    names = []
    count = 0

    for org_id, name in database.get_organizations_without_embeddings(batch_size=batch_size, source=source):
        org_ids.append(org_id)
        names.append(name)

        if len(names) >= batch_size:
            # Generate embeddings
            embeddings = embedder.embed_batch(names)
            database.insert_embeddings_batch(org_ids, embeddings)
            count += len(names)
            click.echo(f"Repaired {count:,} / {missing_count:,} embeddings...", err=True)
            org_ids = []
            names = []

    # Final batch
    if names:
        embeddings = embedder.embed_batch(names)
        database.insert_embeddings_batch(org_ids, embeddings)
        count += len(names)

    click.echo(f"\nRepaired {count:,} embeddings successfully.", err=True)
    database.close()


@db_cmd.command("migrate")
@click.argument("db_path", type=click.Path(exists=True))
@click.option("--rename-file", is_flag=True, help="Also rename companies.db to entities.db")
@click.option("--yes", is_flag=True, help="Skip confirmation prompt")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
def db_migrate(db_path: str, rename_file: bool, yes: bool, verbose: bool):
    """
    Migrate database from legacy schema to new schema.

    Migrates from old naming (companies/company_embeddings tables)
    to new naming (organizations/organization_embeddings tables).

    \b
    What this does:
    - Renames 'companies' table to 'organizations'
    - Renames 'company_embeddings' table to 'organization_embeddings'
    - Updates all indexes

    \b
    Examples:
        corp-extractor db migrate companies.db
        corp-extractor db migrate companies.db --rename-file
        corp-extractor db migrate ~/.cache/corp-extractor/companies.db --yes
    """
    _configure_logging(verbose)

    from pathlib import Path
    from .database import OrganizationDatabase

    db_path_obj = Path(db_path)

    if not yes:
        click.confirm(
            f"This will migrate {db_path} from legacy schema (companies) to new schema (organizations).\n"
            "This operation cannot be undone. Continue?",
            abort=True
        )

    try:
        database = OrganizationDatabase(db_path=db_path)
        migrations = database.migrate_from_legacy_schema()
        database.close()

        if migrations:
            click.echo("Migration completed:")
            for table, action in migrations.items():
                click.echo(f"  {table}: {action}")
        else:
            click.echo("No migration needed. Database already uses new schema.")

        # Optionally rename the file
        if rename_file and db_path_obj.name.startswith("companies"):
            new_name = db_path_obj.name.replace("companies", "entities")
            new_path = db_path_obj.parent / new_name
            db_path_obj.rename(new_path)
            click.echo(f"Renamed file: {db_path} -> {new_path}")

    except Exception as e:
        raise click.ClickException(f"Migration failed: {e}")


# =============================================================================
# Document commands
# =============================================================================

@main.group("document")
def document_cmd():
    """
    Process documents with chunking, deduplication, and citations.

    \b
    Commands:
        process    Process a document through the full pipeline
        chunk      Preview chunking without extraction

    \b
    Examples:
        corp-extractor document process article.txt
        corp-extractor document process report.pdf --no-summary
        corp-extractor document chunk article.txt --max-tokens 500
    """
    pass


@document_cmd.command("process")
@click.argument("input_source")  # Can be file path or URL
@click.option("--title", type=str, help="Document title (for citations)")
@click.option("--author", "authors", type=str, multiple=True, help="Document author(s)")
@click.option("--year", type=int, help="Publication year")
@click.option("--max-tokens", type=int, default=1000, help="Target tokens per chunk (default: 1000)")
@click.option("--overlap", type=int, default=100, help="Token overlap between chunks (default: 100)")
@click.option("--no-summary", is_flag=True, help="Skip document summarization")
@click.option("--no-dedup", is_flag=True, help="Skip deduplication across chunks")
@click.option("--use-ocr", is_flag=True, help="Force OCR for PDF parsing")
@click.option(
    "--stages",
    type=str,
    default="1-6",
    help="Pipeline stages to run (e.g., '1-3' or '1,2,5')"
)
@click.option(
    "-o", "--output",
    type=click.Choice(["table", "json", "triples"], case_sensitive=False),
    default="table",
    help="Output format (default: table)"
)
@click.option("-v", "--verbose", is_flag=True, help="Show verbose output")
@click.option("-q", "--quiet", is_flag=True, help="Suppress progress messages")
def document_process(
    input_source: str,
    title: Optional[str],
    authors: tuple[str, ...],
    year: Optional[int],
    max_tokens: int,
    overlap: int,
    no_summary: bool,
    no_dedup: bool,
    use_ocr: bool,
    stages: str,
    output: str,
    verbose: bool,
    quiet: bool,
):
    """
    Process a document or URL through the extraction pipeline with chunking.

    Supports text files, URLs (web pages and PDFs).

    \b
    Examples:
        corp-extractor document process article.txt
        corp-extractor document process report.txt --title "Annual Report" --year 2024
        corp-extractor document process https://example.com/article
        corp-extractor document process https://example.com/report.pdf --use-ocr
        corp-extractor document process doc.txt --no-summary --stages 1-3
        corp-extractor document process doc.txt -o json
    """
    _configure_logging(verbose)

    # Import document pipeline
    from .document import DocumentPipeline, DocumentPipelineConfig, Document
    from .models.document import ChunkingConfig
    from .pipeline import PipelineConfig
    _load_all_plugins()

    # Parse stages
    enabled_stages = _parse_stages(stages)

    # Build configs
    chunking_config = ChunkingConfig(
        target_tokens=max_tokens,
        max_tokens=max_tokens * 2,
        overlap_tokens=overlap,
    )

    pipeline_config = PipelineConfig(
        enabled_stages=enabled_stages,
    )

    doc_config = DocumentPipelineConfig(
        chunking=chunking_config,
        generate_summary=not no_summary,
        deduplicate_across_chunks=not no_dedup,
        pipeline_config=pipeline_config,
    )

    # Create pipeline
    pipeline = DocumentPipeline(doc_config)

    # Detect if input is a URL
    is_url = input_source.startswith(("http://", "https://"))

    # Process
    try:
        if is_url:
            # Process URL
            from .document import URLLoaderConfig

            if not quiet:
                click.echo(f"Fetching URL: {input_source}", err=True)

            loader_config = URLLoaderConfig(use_ocr=use_ocr)
            ctx = pipeline.process_url_sync(input_source, loader_config)

            if not quiet:
                click.echo(f"Processed: {ctx.document.metadata.title or 'Untitled'}", err=True)

        else:
            # Process file
            from pathlib import Path
            import os

            if not os.path.exists(input_source):
                raise click.ClickException(f"File not found: {input_source}")

            # Read input file
            with open(input_source, "r", encoding="utf-8") as f:
                text = f.read()

            if not text.strip():
                raise click.ClickException("Input file is empty")

            if not quiet:
                click.echo(f"Processing document: {input_source} ({len(text)} chars)", err=True)

            # Create document with metadata
            doc_title = title or Path(input_source).stem
            document = Document.from_text(
                text=text,
                title=doc_title,
                source_type="text",
                authors=list(authors),
                year=year,
            )

            ctx = pipeline.process(document)

        # Output results
        if output == "json":
            _print_document_json(ctx)
        elif output == "triples":
            _print_document_triples(ctx)
        else:
            _print_document_table(ctx, verbose)

        # Report stats
        if not quiet:
            click.echo(f"\nChunks: {ctx.chunk_count}", err=True)
            click.echo(f"Statements: {ctx.statement_count}", err=True)
            if ctx.duplicates_removed > 0:
                click.echo(f"Duplicates removed: {ctx.duplicates_removed}", err=True)

            if ctx.processing_errors:
                click.echo(f"\nErrors: {len(ctx.processing_errors)}", err=True)
                for error in ctx.processing_errors:
                    click.echo(f"  - {error}", err=True)

    except Exception as e:
        logging.exception("Document processing error:")
        raise click.ClickException(f"Processing failed: {e}")


@document_cmd.command("chunk")
@click.argument("input_path", type=click.Path(exists=True))
@click.option("--max-tokens", type=int, default=1000, help="Target tokens per chunk (default: 1000)")
@click.option("--overlap", type=int, default=100, help="Token overlap between chunks (default: 100)")
@click.option("-o", "--output", type=click.Choice(["table", "json"]), default="table", help="Output format")
@click.option("-v", "--verbose", is_flag=True, help="Show verbose output")
def document_chunk(
    input_path: str,
    max_tokens: int,
    overlap: int,
    output: str,
    verbose: bool,
):
    """
    Preview document chunking without running extraction.

    Shows how a document would be split into chunks for processing.

    \b
    Examples:
        corp-extractor document chunk article.txt
        corp-extractor document chunk article.txt --max-tokens 500
        corp-extractor document chunk article.txt -o json
    """
    _configure_logging(verbose)

    # Read input file
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    if not text.strip():
        raise click.ClickException("Input file is empty")

    click.echo(f"Chunking document: {input_path} ({len(text)} chars)", err=True)

    from .document import DocumentChunker, Document
    from .models.document import ChunkingConfig

    config = ChunkingConfig(
        target_tokens=max_tokens,
        max_tokens=max_tokens * 2,
        overlap_tokens=overlap,
    )

    from pathlib import Path
    document = Document.from_text(text, title=Path(input_path).stem)
    chunker = DocumentChunker(config)
    chunks = chunker.chunk_document(document)

    if output == "json":
        import json
        chunk_data = [
            {
                "index": c.chunk_index,
                "tokens": c.token_count,
                "chars": len(c.text),
                "pages": c.page_numbers,
                "overlap": c.overlap_chars,
                "preview": c.text[:100] + "..." if len(c.text) > 100 else c.text,
            }
            for c in chunks
        ]
        click.echo(json.dumps({"chunks": chunk_data, "total": len(chunks)}, indent=2))
    else:
        click.echo(f"\nCreated {len(chunks)} chunk(s):\n")
        click.echo("-" * 80)

        for chunk in chunks:
            click.echo(f"Chunk {chunk.chunk_index + 1}:")
            click.echo(f"  Tokens: {chunk.token_count}")
            click.echo(f"  Characters: {len(chunk.text)}")
            if chunk.page_numbers:
                click.echo(f"  Pages: {chunk.page_numbers}")
            if chunk.overlap_chars > 0:
                click.echo(f"  Overlap: {chunk.overlap_chars} chars")

            preview = chunk.text[:200].replace("\n", " ")
            if len(chunk.text) > 200:
                preview += "..."
            click.echo(f"  Preview: {preview}")
            click.echo("-" * 80)


def _print_document_json(ctx):
    """Print document context as JSON."""
    import json
    click.echo(json.dumps(ctx.as_dict(), indent=2, default=str))


def _print_document_triples(ctx):
    """Print document statements as triples."""
    for stmt in ctx.labeled_statements:
        parts = [stmt.subject_fqn, stmt.statement.predicate, stmt.object_fqn]
        if stmt.page_number:
            parts.append(f"p.{stmt.page_number}")
        click.echo("\t".join(parts))


def _print_document_table(ctx, verbose: bool):
    """Print document context in table format."""
    # Show summary if available
    if ctx.document.summary:
        click.echo("\nDocument Summary:")
        click.echo("-" * 40)
        click.echo(ctx.document.summary)
        click.echo("-" * 40)

    if not ctx.labeled_statements:
        click.echo("\nNo statements extracted.")
        return

    click.echo(f"\nExtracted {len(ctx.labeled_statements)} statement(s):\n")
    click.echo("-" * 80)

    for i, stmt in enumerate(ctx.labeled_statements, 1):
        click.echo(f"{i}. {stmt.subject_fqn}")
        click.echo(f"   --[{stmt.statement.predicate}]-->")
        click.echo(f"   {stmt.object_fqn}")

        # Show citation
        if stmt.citation:
            click.echo(f"   Citation: {stmt.citation}")
        elif stmt.page_number:
            click.echo(f"   Page: {stmt.page_number}")

        # Show labels
        for label in stmt.labels:
            if isinstance(label.label_value, float):
                click.echo(f"   {label.label_type}: {label.label_value:.3f}")
            else:
                click.echo(f"   {label.label_type}: {label.label_value}")

        # Show taxonomy (top 3)
        if stmt.taxonomy_results:
            sorted_taxonomy = sorted(stmt.taxonomy_results, key=lambda t: t.confidence, reverse=True)[:3]
            taxonomy_strs = [f"{t.category}:{t.label}" for t in sorted_taxonomy]
            click.echo(f"   Topics: {', '.join(taxonomy_strs)}")

        if verbose and stmt.statement.source_text:
            source = stmt.statement.source_text[:60] + "..." if len(stmt.statement.source_text) > 60 else stmt.statement.source_text
            click.echo(f"   Source: \"{source}\"")

        click.echo("-" * 80)

    # Show timings in verbose mode
    if verbose and ctx.stage_timings:
        click.echo("\nStage timings:")
        for stage, duration in ctx.stage_timings.items():
            click.echo(f"  {stage}: {duration:.3f}s")


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

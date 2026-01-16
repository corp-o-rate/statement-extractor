"""
Command-line interface for statement extraction.

Usage:
    corp-extractor "Your text here"
    corp-extractor -f input.txt
    cat input.txt | corp-extractor -
"""

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
    ]:
        logging.getLogger(logger_name).setLevel(level)

from . import __version__
from .models import (
    ExtractionOptions,
    PredicateComparisonConfig,
    PredicateTaxonomy,
    ScoringConfig,
)


@click.command()
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
        raise click.UsageError(
            "No input provided. Use: statement-extractor \"text\", "
            "statement-extractor -f file.txt, or pipe via stdin."
        )

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

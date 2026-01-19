# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Corp-Extractor (`corp-extractor` on PyPI) is a Python library that extracts structured subject-predicate-object statements from unstructured text using T5-Gemma2 and GLiNER2 models.

## Commands

```bash
uv sync                              # Install dependencies
uv run pytest                        # Run all tests
uv run pytest tests/test_pipeline.py # Run single test file
uv run pytest -m "not slow"          # Skip slow tests (model loading)
uv run pytest -k "test_name"         # Run specific test by name

uv run ruff check .                  # Lint
uv run mypy src/                     # Type check

uv build                             # Build package
uv publish                           # Publish to PyPI

# CLI testing
uv run corp-extractor split "text"           # Simple extraction (Stage 1)
uv run corp-extractor pipeline "text"        # Full 6-stage pipeline
uv run corp-extractor pipeline "text" -v     # Verbose with debug logs
uv run corp-extractor plugins list           # List registered plugins
```

## Architecture

### 6-Stage Pipeline

The extraction pipeline processes text through sequential stages, each with its own plugin type:

| Stage | Plugin Type | Purpose | Interface |
|-------|-------------|---------|-----------|
| 1 | `BaseSplitterPlugin` | Text → `RawTriple[]` | `split(text, ctx)` |
| 2 | `BaseExtractorPlugin` | `RawTriple[]` → `PipelineStatement[]` | `extract(triples, ctx)` |
| 3 | `BaseQualifierPlugin` | Entity → `EntityQualifiers` | `qualify(entity, ctx)` |
| 4 | `BaseCanonicalizerPlugin` | `QualifiedEntity` → `CanonicalMatch` | `find_canonical(entity, ctx)` |
| 5 | `BaseLabelerPlugin` | Statement → `StatementLabel` | `label(stmt, subj, obj, ctx)` |
| 6 | `BaseTaxonomyPlugin` | Statement → `TaxonomyResult[]` | `classify(stmt, subj, obj, ctx)` |

### Plugin Registration

Plugins auto-register via decorators when their modules are imported:

```python
from statement_extractor.pipeline.registry import PluginRegistry

@PluginRegistry.labeler
class MyLabeler(BaseLabelerPlugin):
    @property
    def name(self) -> str:
        return "my_labeler"

    @property
    def label_type(self) -> str:
        return "my_label"
    # ...
```

Plugins are sorted by `priority` property (lower = runs first). Default is 100.

### Key Source Files

- `pipeline/orchestrator.py` - Main pipeline coordinator, runs stages in order
- `pipeline/registry.py` - Plugin registration with `@PluginRegistry.<stage>` decorators
- `pipeline/context.py` - `PipelineContext` that flows through all stages
- `plugins/base.py` - Abstract base classes for all plugin types
- `plugins/extractors/gliner2.py` - GLiNER2 entity/relation extraction (Stage 2)
- `plugins/taxonomy/embedding.py` - Embedding-based taxonomy classification (Stage 6)
- `cli.py` - CLI entry point (`corp-extractor` command)

### Data Models Flow

```
Text → RawTriple → PipelineStatement → QualifiedEntity → CanonicalEntity → LabeledStatement
                   (with ExtractedEntity)                                   (with TaxonomyResult[])
```

Models are defined in `models/` subdirectory: `entity.py`, `statement.py`, `qualifiers.py`, `canonical.py`, `labels.py`.

### GLiNER2 Relation Extraction

The `gliner2_extractor` uses 324 default predicates from `data/default_predicates.json` organized into 21 categories. Relations below `min_confidence` (default 0.75) are filtered out.

### Taxonomy Classification

Stage 6 uses embedding-based classification against `data/statement_taxonomy.json` (ESG topics). Results are stored both in `ctx.taxonomy_results` and on each `LabeledStatement.taxonomy_results`.

## Testing

Tests use pytest markers:
- `@pytest.mark.slow` - Tests that load models (skip with `-m "not slow"`)
- `@pytest.mark.pipeline` - Pipeline architecture tests

Fixtures in `conftest.py` provide `sample_source_text` and `sample_statements`.

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
uv run corp-extractor pipeline "text"        # Full 5-stage pipeline
uv run corp-extractor pipeline "text" -v     # Verbose with debug logs
uv run corp-extractor plugins list           # List registered plugins

# Entity database commands
uv run corp-extractor db status              # Show database stats
uv run corp-extractor db search "Microsoft"  # Search for organization
uv run corp-extractor db search-people "Tim Cook"  # Search for person (v0.9.0)
uv run corp-extractor db import-gleif --download --limit 10000  # Import GLEIF (~3M records)
uv run corp-extractor db import-sec --download                  # Import SEC bulk (~100K+ filers)
uv run corp-extractor db import-companies-house --download --limit 10000  # Import UK companies
uv run corp-extractor db import-wikidata --limit 5000  # Import Wikidata orgs
uv run corp-extractor db import-people --type executive --limit 5000  # Import notable people (v0.9.0)
uv run corp-extractor db import-people --all --limit 10000            # All person types (v0.9.0)
uv run corp-extractor db upload              # Upload with lite/compressed variants
uv run corp-extractor db download            # Download lite version (default)
uv run corp-extractor db download --full     # Download full version
uv run corp-extractor db create-lite entities.db  # Create lite version locally
uv run corp-extractor db compress entities.db     # Compress with gzip

# Document processing commands
uv run corp-extractor document process article.txt
uv run corp-extractor document process https://example.com/article
uv run corp-extractor document process report.pdf --use-ocr
uv run corp-extractor document chunk article.txt --max-tokens 500
```

## Architecture

### 5-Stage Pipeline

The extraction pipeline processes text through sequential stages, each with its own plugin type:

| Stage | Plugin Type | Purpose | Interface |
|-------|-------------|---------|-----------|
| 1 | `BaseSplitterPlugin` | Text → `RawTriple[]` | `split(text, ctx)` |
| 2 | `BaseExtractorPlugin` | `RawTriple[]` → `PipelineStatement[]` | `extract(triples, ctx)` |
| 3 | `BaseQualifierPlugin` | Entity → `CanonicalEntity` | `qualify(entity, ctx)` |
| 4 | `BaseLabelerPlugin` | Statement → `StatementLabel` | `label(stmt, subj, obj, ctx)` |
| 5 | `BaseTaxonomyPlugin` | Statement → `TaxonomyResult[]` | `classify(stmt, subj, obj, ctx)` |

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
- `plugins/taxonomy/embedding.py` - Embedding-based taxonomy classification (Stage 5)
- `cli.py` - CLI entry point (`corp-extractor` command)

### Entity Database Module

The `database/` module provides organization and person embedding storage and search:

- `database/models.py` - `CompanyRecord`, `CompanyMatch`, `PersonRecord`, `PersonMatch`, `PersonType`, `DatabaseStats`, `EntityType` Pydantic models
- `database/store.py` - `OrganizationDatabase` and `PersonDatabase` SQLite+sqlite-vec storage
- `database/embeddings.py` - `CompanyEmbedder` using google/embeddinggemma-300m
- `database/hub.py` - HuggingFace Hub upload/download with lite/compressed variants
- `database/resolver.py` - `OrganizationResolver` shared utility for org lookups (used by person.py and embedding_company.py)
- `database/importers/` - Data source importers:
  - `gleif.py` - GLEIF LEI data (XML/JSON, ~3M records) - maps EntityCategory to EntityType
  - `sec_edgar.py` - SEC bulk submissions.zip (~100K+ filers) - maps SIC codes to EntityType
  - `companies_house.py` - UK Companies House bulk data (~5M records) - maps company_type to EntityType
  - `wikidata.py` - Wikidata SPARQL queries (35+ entity types) - maps query types to EntityType
  - `wikidata_people.py` - Wikidata SPARQL queries for notable people (executives, politicians, athletes, etc.)

**EntityType Classification:**
Each organization record is classified with an `entity_type` field:
- Business: `business`, `fund`, `branch`
- Non-profit: `nonprofit`, `ngo`, `foundation`, `trade_union`
- Government: `government`, `international_org`, `political_party`
- Other: `educational`, `research`, `healthcare`, `media`, `sports`, `religious`, `unknown`

**PersonType Classification (v0.9.0):**
Each person record is classified with a `person_type` field:
- `executive` - CEOs, board members, C-suite
- `politician` - Elected officials, diplomats
- `academic` - Professors, researchers
- `artist` - Musicians, actors, directors, writers
- `athlete` - Sports figures
- `entrepreneur` - Founders, business owners
- `journalist` - Reporters, media personalities
- `activist` - Advocates, campaigners
- `scientist` - Scientists, inventors
- `unknown` - Type not determined

**Database variants:**
- `entities.db` - Full database with complete record metadata
- `entities-lite.db` - Lite version without record data (default download, smaller)
- `.gz` compressed versions for efficient transfer

### Document Processing Module

The `document/` module provides document-level extraction:

- `document/chunker.py` - Token-based text chunking with overlap
- `document/context.py` - `DocumentContext` for tracking extraction state
- `document/deduplicator.py` - Cross-chunk statement deduplication
- `document/html_extractor.py` - HTML content extraction (Readability-style)
- `document/loader.py` - URL and file loading with content type detection
- `document/pipeline.py` - `DocumentPipeline` orchestrator
- `document/summarizer.py` - Document summarization

**PDF and Scraper Plugins:**
- `plugins/pdf/pypdf.py` - PDF parsing with PyMuPDF (optional OCR with pytesseract)
- `plugins/scrapers/http.py` - HTTP/URL scraping with httpx

### Data Models Flow

```
Text → RawTriple → PipelineStatement → QualifiedEntity → CanonicalEntity → LabeledStatement
                   (with ExtractedEntity)                                   (with TaxonomyResult[])
```

Models are defined in `models/` subdirectory: `entity.py`, `statement.py`, `qualifiers.py`, `canonical.py`, `labels.py`.

### GLiNER2 Relation Extraction

The `gliner2_extractor` uses 324 default predicates from `data/default_predicates.json` organized into 21 categories. Relations below `min_confidence` (default 0.75) are filtered out.

### Taxonomy Classification

Stage 5 uses embedding-based classification against `data/statement_taxonomy.json` (ESG topics). Results are stored both in `ctx.taxonomy_results` and on each `LabeledStatement.taxonomy_results`.

## Testing

Tests use pytest markers:
- `@pytest.mark.slow` - Tests that load models (skip with `-m "not slow"`)
- `@pytest.mark.pipeline` - Pipeline architecture tests

Fixtures in `conftest.py` provide `sample_source_text` and `sample_statements`.

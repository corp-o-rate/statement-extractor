---
license: mit
task_categories:
  - text-classification
  - feature-extraction
language:
  - en
tags:
  - entity-resolution
  - named-entity-recognition
  - company-data
  - embeddings
  - sqlite
  - vector-search
size_categories:
  - 1M<n<10M
---

# Entity References Database

A pre-built SQLite database with vector embeddings for organization/entity resolution. Used by the [corp-extractor](https://pypi.org/project/corp-extractor/) library for fast entity qualification via embedding similarity search.

## Overview

This database contains organization records from multiple authoritative sources, each with:
- Organization name (canonical)
- Source identifier (LEI, CIK, UK Company Number, Wikidata QID)
- Entity type classification (business, nonprofit, government, etc.)
- Vector embeddings for semantic search (768-dim, using [embeddinggemma-300m](https://huggingface.co/google/embeddinggemma-300m))

## Data Sources

| Source | Records | Identifier | Description |
|--------|---------|------------|-------------|
| GLEIF | ~3.2M | LEI (Legal Entity Identifier) | Global legal entities from the LEI system |
| SEC Edgar | ~100K+ | CIK (Central Index Key) | US SEC-registered filers |
| Companies House | ~5M | UK Company Number | UK registered companies |
| Wikidata | Variable | Wikidata QID | Notable organizations from Wikidata |

## Entity Types

Organizations are classified into the following types:

| Category | Types |
|----------|-------|
| Business | `business`, `fund`, `branch` |
| Non-profit | `nonprofit`, `ngo`, `foundation`, `trade_union` |
| Government | `government`, `international_org`, `political_party` |
| Other | `educational`, `research`, `healthcare`, `media`, `sports`, `religious`, `unknown` |

## Database Variants

| File | Description | Use Case |
|------|-------------|----------|
| `entities.db` | Full database with complete source record metadata | When you need full record details |
| `entities-lite.db` | Lite version without record data | Default - faster download, smaller size |
| `entities.db.gz` | Compressed full database | When bandwidth is limited |
| `entities-lite.db.gz` | Compressed lite database | Smallest download size |

## Schema

### organizations table
```sql
CREATE TABLE organizations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    name_normalized TEXT NOT NULL,
    source TEXT NOT NULL,  -- 'gleif', 'sec_edgar', 'companies_house', 'wikipedia'
    source_id TEXT NOT NULL,
    region TEXT NOT NULL DEFAULT '',
    entity_type TEXT NOT NULL DEFAULT 'unknown',
    record TEXT NOT NULL,  -- JSON with full source record (empty in lite version)
    UNIQUE(source, source_id)
);
```

### organization_embeddings table (sqlite-vec)
```sql
CREATE VIRTUAL TABLE organization_embeddings USING vec0(
    org_id INTEGER PRIMARY KEY,
    embedding float[768]
);
```

## Usage with corp-extractor

```bash
# Install
pip install corp-extractor

# Download the database (lite version by default)
corp-extractor db download

# Download full version
corp-extractor db download --full

# Search for an organization
corp-extractor db search "Microsoft"

# Check database status
corp-extractor db status
```

### Python API

```python
from statement_extractor.database import OrganizationDatabase, CompanyEmbedder

# Load database
database = OrganizationDatabase()
embedder = CompanyEmbedder()

# Search by embedding similarity
query_embedding = embedder.embed("Microsoft Corporation")
results = database.search(query_embedding, top_k=5)

for record, similarity in results:
    print(f"{record.name} ({record.source}:{record.source_id}) - {similarity:.3f}")
```

## Building Your Own Database

```bash
# Import from authoritative sources
corp-extractor db import-gleif --download
corp-extractor db import-sec --download
corp-extractor db import-companies-house --download
corp-extractor db import-wikidata --limit 50000

# Upload to HuggingFace
export HF_TOKEN="hf_..."
corp-extractor db upload
```

## License

MIT License - the database structure and embedding generation code are MIT licensed.

Individual data sources have their own licenses:
- GLEIF: Open license for LEI data
- SEC Edgar: Public domain (US government)
- Companies House: Open Government Licence
- Wikidata: CC0 (public domain)

## Links

- [Corp-Extractor on PyPI](https://pypi.org/project/corp-extractor/)
- [Corp-Extractor GitHub](https://github.com/corp-o-rate/statement-extractor)
- [Statement Extractor Model](https://huggingface.co/Corp-o-Rate-Community/statement-extractor)

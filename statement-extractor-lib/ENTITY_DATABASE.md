# Entity Database Guide

The entity database provides fast lookup and qualification of organizations and people using vector similarity search. It stores records from authoritative sources (GLEIF, SEC, Companies House, Wikidata) with 768-dimensional embeddings for semantic matching.

## Quick Start

```bash
# Download the pre-built database (recommended)
corp-extractor db download

# Check what's in it
corp-extractor db status

# Search for organizations
corp-extractor db search "Microsoft"

# Search for people
corp-extractor db search-people "Tim Cook"
```

The database is automatically used by the pipeline's qualification stage (Stage 3) to resolve entity names to canonical identifiers.

## Getting the Database

### Download Pre-built Database

The fastest way to get started is downloading the pre-built database from HuggingFace:

```bash
# Download lite version (default, smaller, faster)
corp-extractor db download

# Download full version (includes complete source metadata)
corp-extractor db download --full
```

**Database variants:**

| File | Size | Contents |
|------|------|----------|
| `entities-lite.db` | ~500MB | Core fields + embeddings only |
| `entities.db` | ~2GB | Full records with source metadata |

The lite version is recommended for most use cases. It contains all the information needed for entity resolution.

**Storage location:** `~/.cache/corp-extractor/entities.db`

**HuggingFace repo:** [Corp-o-Rate-Community/entity-references](https://huggingface.co/datasets/Corp-o-Rate-Community/entity-references)

### Automatic Download

If you use the pipeline without downloading first, the database is downloaded automatically on first use:

```python
from statement_extractor.pipeline import ExtractionPipeline

pipeline = ExtractionPipeline()
ctx = pipeline.process("Microsoft CEO Satya Nadella announced...")
# Database downloaded automatically if not present
```

## Database Format

The database uses SQLite with the [sqlite-vec](https://github.com/asg017/sqlite-vec) extension for vector similarity search.

### Schema

**Organizations table:**
```sql
CREATE TABLE organizations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    name_normalized TEXT NOT NULL,
    source TEXT NOT NULL,        -- 'gleif', 'sec_edgar', 'companies_house', 'wikipedia'
    source_id TEXT NOT NULL,     -- LEI, CIK, UK Company Number, or QID
    region TEXT NOT NULL DEFAULT '',
    entity_type TEXT NOT NULL DEFAULT 'unknown',
    from_date TEXT,              -- ISO YYYY-MM-DD (inception/registration)
    to_date TEXT,                -- ISO YYYY-MM-DD (dissolution)
    record TEXT NOT NULL,        -- JSON (empty in lite version)
    UNIQUE(source, source_id)
);

CREATE VIRTUAL TABLE organization_embeddings USING vec0(
    org_id INTEGER PRIMARY KEY,
    embedding float[768]
);
```

**People table:**
```sql
CREATE TABLE people (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    name_normalized TEXT NOT NULL,
    source TEXT NOT NULL,        -- 'wikidata'
    source_id TEXT NOT NULL,     -- Wikidata QID
    country TEXT NOT NULL DEFAULT '',
    person_type TEXT NOT NULL DEFAULT 'unknown',
    known_for_role TEXT DEFAULT '',
    known_for_org TEXT DEFAULT '',
    known_for_org_id INTEGER,    -- FK to organizations table
    from_date TEXT,              -- Role start date (ISO)
    to_date TEXT,                -- Role end date (ISO)
    birth_date TEXT,             -- ISO YYYY-MM-DD
    death_date TEXT,             -- ISO YYYY-MM-DD
    record TEXT NOT NULL,
    UNIQUE(source, source_id, known_for_role, known_for_org)
);

CREATE VIRTUAL TABLE person_embeddings USING vec0(
    person_id INTEGER PRIMARY KEY,
    embedding float[768]
);
```

### Entity Types

**Organization EntityTypes:**

| Category | Types |
|----------|-------|
| Business | `business`, `fund`, `branch` |
| Non-profit | `nonprofit`, `ngo`, `foundation`, `trade_union` |
| Government | `government`, `international_org`, `political_party` |
| Education | `educational`, `research` |
| Other | `healthcare`, `media`, `sports`, `religious`, `unknown` |

**Person PersonTypes:**

| Type | Description | Examples |
|------|-------------|----------|
| `executive` | C-suite, board members | Tim Cook, Satya Nadella |
| `politician` | Elected officials | Presidents, MPs, mayors |
| `government` | Civil servants, diplomats | Agency heads, ambassadors |
| `military` | Armed forces personnel | Generals, admirals |
| `legal` | Judges, lawyers | Supreme Court justices |
| `professional` | Known for profession | Famous surgeons, architects |
| `academic` | Professors, researchers | Neil deGrasse Tyson |
| `scientist` | Scientists, inventors | Research scientists |
| `athlete` | Sports figures | LeBron James |
| `artist` | Traditional creatives | Musicians, actors, painters |
| `media` | Internet personalities | YouTubers, influencers |
| `journalist` | Reporters, presenters | Anderson Cooper |
| `entrepreneur` | Founders, business owners | Mark Zuckerberg |
| `activist` | Advocates, campaigners | Greta Thunberg |

### Data Sources

**Organizations:**

| Source | Records | Identifier | Coverage |
|--------|---------|------------|----------|
| GLEIF | ~3.2M | LEI (Legal Entity Identifier) | Global legal entities |
| SEC Edgar | ~100K+ | CIK (Central Index Key) | US public companies |
| Companies House | ~5M | UK Company Number | UK registered companies |
| Wikidata | Variable | QID | Notable organizations |

**People:**

| Source | Records | Identifier | Coverage |
|--------|---------|------------|----------|
| Wikidata | Variable | QID | Notable people worldwide |

### Embedding Model

Embeddings are generated using [google/embeddinggemma-300m](https://huggingface.co/google/embeddinggemma-300m) (300M parameters, 768 dimensions). This model is optimized for CPU inference and downloaded automatically on first use.

## CLI Reference

### Database Status

```bash
# Show record counts and database info
corp-extractor db status
```

Output:
```
Database: /Users/you/.cache/corp-extractor/entities.db
Organizations: 3,245,891 records
  - gleif: 3,200,000
  - sec_edgar: 12,500
  - companies_house: 33,391
People: 50,000 records
  - wikidata: 50,000
```

### Search

```bash
# Search organizations
corp-extractor db search "Microsoft"
corp-extractor db search "Apple Inc" --top 10

# Search people
corp-extractor db search-people "Tim Cook"
corp-extractor db search-people "Elon Musk" --top 5
```

Search uses hybrid matching: exact name lookup combined with embedding similarity.

### Import Data

**Organizations:**

```bash
# GLEIF - Global LEI data (~3M records)
corp-extractor db import-gleif --download
corp-extractor db import-gleif /path/to/lei-data.json --limit 50000

# SEC Edgar - US public companies (~100K+ filers)
corp-extractor db import-sec --download

# UK Companies House (~5M records)
corp-extractor db import-companies-house --download
corp-extractor db import-companies-house --download --limit 100000

# Wikidata organizations via SPARQL
corp-extractor db import-wikidata --limit 50000
```

**People:**

```bash
# Import by person type
corp-extractor db import-people --type executive --limit 5000
corp-extractor db import-people --type politician --limit 5000
corp-extractor db import-people --type athlete --limit 5000

# Import all person types
corp-extractor db import-people --all --limit 50000

# Skip existing records (faster for incremental updates)
corp-extractor db import-people --type executive --skip-existing

# Fetch role start/end dates (slower, queries per person)
corp-extractor db import-people --type executive --enrich-dates
```

**Wikidata Dump Import (v0.9.1):**

For large imports without SPARQL query timeouts:

```bash
# Download and import from Wikidata dump (~100GB compressed)
corp-extractor db import-wikidata-dump --download --limit 100000

# Import from local dump file
corp-extractor db import-wikidata-dump --dump /path/to/latest-all.json.bz2

# Import only people (no organizations)
corp-extractor db import-wikidata-dump --dump dump.bz2 --people --no-orgs

# Import only organizations (no people)
corp-extractor db import-wikidata-dump --dump dump.bz2 --orgs --no-people

# Resume interrupted import
corp-extractor db import-wikidata-dump --dump dump.bz2 --resume

# Skip records already in database
corp-extractor db import-wikidata-dump --dump dump.bz2 --skip-updates

# Only organizations with English Wikipedia articles
corp-extractor db import-wikidata-dump --dump dump.bz2 --require-enwiki
```

The dump importer tracks progress and supports resume via `~/.cache/corp-extractor/wikidata-dump-progress.json`.

### Canonicalization

Link equivalent records across sources:

```bash
corp-extractor db canonicalize
```

This matches records by:
- Global identifiers (LEI, CIK, ticker)
- Normalized name + region

Source priority: gleif > sec_edgar > companies_house > wikipedia

### Download and Upload

```bash
# Download from HuggingFace
corp-extractor db download                   # Lite version (default)
corp-extractor db download --full            # Full version

# Upload to HuggingFace (requires HF_TOKEN)
export HF_TOKEN="hf_..."
corp-extractor db upload                     # Upload all variants
corp-extractor db upload --no-lite           # Skip lite version
corp-extractor db upload --no-compress       # Skip compressed versions

# Create lite version locally
corp-extractor db create-lite entities.db
```

### Database Maintenance

```bash
# Repair missing embeddings
corp-extractor db repair-embeddings

# Migrate/rebuild schema
corp-extractor db migrate
```

## Python API

### Search Organizations

```python
from statement_extractor.database import OrganizationDatabase

db = OrganizationDatabase()

# Search by name (hybrid: text + embedding)
matches = db.search_by_name("Microsoft Corporation", top_k=5)
for match in matches:
    print(f"{match.company.name} ({match.company.source}:{match.company.source_id})")
    print(f"  Similarity: {match.similarity_score:.3f}")
    print(f"  Type: {match.company.entity_type}")

# Search by embedding
from statement_extractor.database import CompanyEmbedder

embedder = CompanyEmbedder()
embedding = embedder.embed("Microsoft")
matches = db.search(embedding, top_k=10, min_similarity=0.7)
```

### Search People

```python
from statement_extractor.database import PersonDatabase

db = PersonDatabase()

# Search by name
matches = db.search_by_name("Tim Cook", top_k=5)
for match in matches:
    print(f"{match.person.name} - {match.person.known_for_role} at {match.person.known_for_org}")
    print(f"  Wikidata: {match.person.source_id}")
    print(f"  Type: {match.person.person_type}")
```

### Use in Pipeline

The database is automatically used by qualification plugins:

```python
from statement_extractor.pipeline import ExtractionPipeline

pipeline = ExtractionPipeline()
ctx = pipeline.process("Microsoft CEO Satya Nadella announced new AI features.")

for stmt in ctx.labeled_statements:
    print(f"{stmt.subject_fqn} --[{stmt.statement.predicate}]--> {stmt.object_fqn}")
    # e.g., "Satya Nadella (CEO, Microsoft) --[announced]--> new AI features"
```

### Add Custom Records

```python
from statement_extractor.database import OrganizationDatabase, CompanyRecord, EntityType

db = OrganizationDatabase()

record = CompanyRecord(
    name="My Company Inc",
    source="custom",
    source_id="CUSTOM001",
    region="US",
    entity_type=EntityType.business,
    record={"custom_field": "value"},
)
db.add_record(record)
```

### Database Statistics

```python
from statement_extractor.database import OrganizationDatabase, PersonDatabase

org_db = OrganizationDatabase()
person_db = PersonDatabase()

org_stats = org_db.get_stats()
print(f"Total organizations: {org_stats.total_records}")
for source, count in org_stats.records_by_source.items():
    print(f"  {source}: {count}")

person_stats = person_db.get_stats()
print(f"Total people: {person_stats.total_records}")
```

## Building Your Own Database

### Full Build Process

```bash
# 1. Import from all sources
corp-extractor db import-gleif --download
corp-extractor db import-sec --download
corp-extractor db import-companies-house --download
corp-extractor db import-wikidata --limit 100000
corp-extractor db import-wikidata-dump --download --people --no-orgs --limit 100000

# 2. Link equivalent records
corp-extractor db canonicalize

# 3. Check status
corp-extractor db status

# 4. Upload to HuggingFace
export HF_TOKEN="hf_..."
corp-extractor db upload
```

### Incremental Updates

```bash
# Add new records without re-importing everything
corp-extractor db import-people --type executive --skip-existing
corp-extractor db import-wikidata-dump --dump dump.bz2 --skip-updates --resume
```

### Custom Database Location

```python
from statement_extractor.database import OrganizationDatabase

# Use custom path
db = OrganizationDatabase("/path/to/my/database.db")
```

Or set environment variable:
```bash
export CORP_EXTRACTOR_DB_PATH="/path/to/my/database.db"
corp-extractor db status
```

## Troubleshooting

### Database Not Found

```
Error: Database not found at ~/.cache/corp-extractor/entities.db
```

Run `corp-extractor db download` to fetch the pre-built database.

### sqlite-vec Extension Error

```
Error: no such module: vec0
```

The sqlite-vec extension should install automatically. If not:
```bash
pip install sqlite-vec
```

### Slow Embedding Generation

Embedding generation is CPU-intensive. For large imports:
- Use `--limit` to test with a subset first
- Consider running on a machine with more cores
- The embedder uses batching automatically

### Resume Interrupted Import

For Wikidata dump imports:
```bash
corp-extractor db import-wikidata-dump --dump dump.bz2 --resume
```

Progress is saved to `~/.cache/corp-extractor/wikidata-dump-progress.json`.

### Memory Issues

The full Wikidata dump is ~100GB compressed. For limited memory:
```bash
# Import in smaller batches
corp-extractor db import-wikidata-dump --dump dump.bz2 --limit 10000 --skip-updates
# Then resume for more
corp-extractor db import-wikidata-dump --dump dump.bz2 --limit 10000 --skip-updates --resume
```

## Data Model Reference

### CompanyRecord

```python
class CompanyRecord(BaseModel):
    name: str                    # Organization name
    source: str                  # 'gleif', 'sec_edgar', 'companies_house', 'wikipedia'
    source_id: str               # LEI, CIK, UK Company Number, or QID
    region: str                  # Country/region code
    entity_type: EntityType      # Classification
    from_date: Optional[str]     # ISO YYYY-MM-DD
    to_date: Optional[str]       # ISO YYYY-MM-DD
    record: dict[str, Any]       # Full source record (empty in lite)

    @property
    def canonical_id(self) -> str:
        return f"{self.source}:{self.source_id}"
```

### PersonRecord

```python
class PersonRecord(BaseModel):
    name: str                    # Display name
    source: str                  # 'wikidata'
    source_id: str               # Wikidata QID
    country: str                 # Country code
    person_type: PersonType      # Classification
    known_for_role: str          # Primary role
    known_for_org: str           # Primary organization name
    known_for_org_id: Optional[int]  # FK to organizations
    from_date: Optional[str]     # Role start (ISO)
    to_date: Optional[str]       # Role end (ISO)
    birth_date: Optional[str]    # Birth date (ISO)
    death_date: Optional[str]    # Death date (ISO)
    record: dict[str, Any]       # Full source record

    @property
    def is_historic(self) -> bool:
        return self.death_date is not None
```

### CompanyMatch / PersonMatch

```python
class CompanyMatch(BaseModel):
    company: CompanyRecord
    similarity_score: float      # 0.0 to 1.0

class PersonMatch(BaseModel):
    person: PersonRecord
    similarity_score: float      # 0.0 to 1.0
    llm_confirmed: bool          # Whether LLM validated match
```

# Entity Embedding Database

The entity embedding database enables fast entity qualification by matching organization and person names against pre-computed embeddings from authoritative sources.

## Overview

The database uses `sqlite-vec` for vector similarity search, storing **two types of records**:

### Organization Records
- **name**: Searchable organization name
- **embedding**: 768-dimensional vector from `google/embeddinggemma-300m`
- **source**: Data source (gleif, sec_edgar, companies_house, wikidata)
- **source_id**: Unique identifier (LEI, CIK, company number, Wikidata QID)
- **record**: Full JSON record from source

### Person Records *(v0.9.0)*
- **name**: Person name (used for embedding and display)
- **embedding**: 768-dimensional vector combining name, role, and organization
- **source**: Data source (wikidata)
- **source_id**: Wikidata QID (e.g., Q312 for Tim Cook)
- **person_type**: Classification (executive, politician, athlete, artist, etc.)
- **known_for_role**: Primary role from Wikipedia (e.g., "CEO", "President")
- **known_for_org**: Primary organization (e.g., "Apple Inc", "Tesla")
- **from_date**: Role start date (ISO format YYYY-MM-DD)
- **to_date**: Role end date (ISO format YYYY-MM-DD)
- **birth_date**: Date of birth (ISO format YYYY-MM-DD) *(v0.9.2)*
- **death_date**: Date of death (ISO format YYYY-MM-DD) - if set, person is historic *(v0.9.2)*
- **record**: Full JSON record from source

**Note:** The same person can have multiple records with different role/org combinations. The unique constraint is on `(source, source_id, known_for_role, known_for_org)`.

**Historic detection:** Person records include an `is_historic` property that returns True for deceased individuals (those with a `death_date` set).

## Data Sources

### Organizations

| Source | Records | Identifier | Date Fields | Coverage |
|--------|---------|------------|-------------|----------|
| [GLEIF](https://www.gleif.org/) | ~3.2M | LEI (Legal Entity Identifier) | `from_date`: LEI registration date | Global companies with LEI |
| [SEC Edgar](https://www.sec.gov/) | ~100K+ | CIK (Central Index Key) | `from_date`: First SEC filing date | All SEC filers (not just public companies) |
| [Companies House](https://www.gov.uk/government/organisations/companies-house) | ~5M | Company Number | `from_date`: Incorporation, `to_date`: Dissolution | UK registered companies |
| [Wikidata](https://www.wikidata.org/) | Variable | QID | `from_date`: Inception (P571), `to_date`: Dissolution (P576) | Notable companies worldwide |

### People *(v0.9.0)*

| Source | Records | Identifier | Date Fields | Coverage |
|--------|---------|------------|-------------|----------|
| [Wikidata](https://www.wikidata.org/) | Variable | QID | `from_date`: Position start (P580), `to_date`: Position end (P582) | Notable people with English Wikipedia articles |
| [SEC Form 4](https://www.sec.gov/) *(v0.9.3)* | ~280K/year | Owner CIK | `from_date`: Period of report | US public company officers, directors, 10%+ owners |
| [Companies House](https://www.gov.uk/government/organisations/companies-house) *(v0.9.3)* | ~15M+ | Person number | `from_date`: Appointment date, `to_date`: Resignation date | UK company officers (directors, secretaries) |

**Person Types:**

| Type | Description | Example Roles |
|------|-------------|---------------|
| `executive` | C-suite, board members | CEO, CFO, Chairman, Director |
| `politician` | Elected officials (presidents, MPs, mayors) | President, Senator, Mayor |
| `government` | Civil servants, diplomats, appointed officials | Ambassador, Agency head |
| `military` | Military officers, armed forces personnel | General, Admiral, Commander |
| `legal` | Judges, lawyers, legal professionals | Supreme Court Justice, Attorney |
| `professional` | Known for profession (doctors, engineers) | Famous surgeon, Architect |
| `athlete` | Sports figures | Players, coaches, team owners |
| `artist` | Traditional creatives (musicians, actors, painters) | Actors, musicians, directors, writers |
| `media` | Internet/social media personalities | YouTuber, Influencer, Podcaster |
| `academic` | Professors, researchers | Professor, Dean, Researcher |
| `scientist` | Scientists, inventors | Research scientist, Lab director |
| `journalist` | Reporters, news presenters | Reporter, Editor, Anchor |
| `entrepreneur` | Founders, business owners | Founder, Co-founder |
| `activist` | Advocates, campaigners | Human rights activist |

## Building the Database

### Prerequisites

```bash
cd statement-extractor-lib
uv sync
```

### Import Data Sources

**1. GLEIF (Global LEI Data)**

```bash
# Download and import latest GLEIF data (3.2M records, ~2GB download)
corp-extractor db import-gleif --download

# Import with limit for testing
corp-extractor db import-gleif --download --limit 100000

# Show available GLEIF file info
corp-extractor db gleif-info
```

**2. SEC Edgar (All SEC Filers)**

```bash
# Download and import bulk SEC filer data (~100K+ records)
corp-extractor db import-sec --download

# Import with limit
corp-extractor db import-sec --download --limit 50000
```

**3. Companies House (UK Companies)**

```bash
# Download and import bulk data (5M records, ~500MB download)
corp-extractor db import-companies-house --download

# Import with limit
corp-extractor db import-companies-house --download --limit 100000

# Alternative: API search (requires free API key)
export COMPANIES_HOUSE_API_KEY="your-key"
corp-extractor db import-companies-house --search "bank,insurance,technology"
```

Get a free API key at: https://developer.company-information.service.gov.uk/

**4. Wikidata (Organizations)**

```bash
# Import companies with LEI codes (default, fastest)
corp-extractor db import-wikidata --limit 50000

# Import specific organization types
corp-extractor db import-wikidata --query-type organization --limit 50000  # All orgs
corp-extractor db import-wikidata --query-type nonprofit --limit 50000     # Non-profits
corp-extractor db import-wikidata --query-type government --limit 50000    # Gov agencies

# Import all organization types (slower, runs all queries)
corp-extractor db import-wikidata --all
```

Available query types: `lei`, `ticker`, `public`, `business`, `organization`, `nonprofit`, `government`

**5. Wikidata People (SPARQL)** *(v0.9.0)*

```bash
# Import notable people (executives by default)
corp-extractor db import-people --type executive --limit 5000

# Import specific person types
corp-extractor db import-people --type politician --limit 5000
corp-extractor db import-people --type athlete --limit 5000
corp-extractor db import-people --type artist --limit 5000

# Import all person types (runs all queries sequentially)
corp-extractor db import-people --all --limit 10000

# Skip existing records instead of updating them
corp-extractor db import-people --type executive --skip-existing

# Enrich people with start/end dates (slower, queries individual records)
corp-extractor db import-people --type executive --enrich-dates
```

Available person types: `executive`, `politician`, `government`, `military`, `legal`, `professional`, `athlete`, `artist`, `media`, `academic`, `scientist`, `journalist`, `entrepreneur`, `activist`

**Note**: Organizations discovered during people import (employers, affiliated orgs) are automatically inserted into the organizations table if they don't already exist. This creates foreign key links via `known_for_org_id`.

**6. SEC Form 4 Officers (US Insiders)** *(v0.9.3)*

```bash
# Import SEC Form 4 officers/directors from 2020 onwards
corp-extractor db import-sec-officers --limit 10000

# Start from a specific year
corp-extractor db import-sec-officers --start-year 2023

# Resume interrupted import
corp-extractor db import-sec-officers --resume

# Skip existing records
corp-extractor db import-sec-officers --skip-existing -v
```

Imports from SEC EDGAR quarterly index files. Each Form 4 filing contains:
- **Issuer**: Company CIK, name, ticker
- **Reporting Owner**: Person name, CIK, relationship (officer/director/10%+ owner)
- **Officer Title**: If officer, their specific title (CEO, CFO, etc.)

Rate limited to 5 requests/second per SEC guidelines. Progress saved for resume capability.

**7. Companies House Officers (UK Directors)** *(v0.9.3)*

```bash
# Import from bulk officers file (requires special request)
corp-extractor db import-ch-officers --file officers.zip --limit 10000

# Resume interrupted import
corp-extractor db import-ch-officers --file officers.zip --resume

# Include resigned officers (default: current only)
corp-extractor db import-ch-officers --file officers.zip --include-resigned
```

**Obtaining the data:**
The officers bulk file (Prod195) is not publicly available. Request access by emailing:
**BulkProducts@companieshouse.gov.uk**

Explain your use case - they typically provide a download link within a few days.

**Data format:**
- Fixed-width format with `<`-delimited variable fields
- 14 fields per officer: Title, Forenames, Surname, Honours, Address (5 fields), Occupation, Nationality, Usual Country
- Position 24 indicates corporate vs individual officers (only individuals imported)
- Approximately 8GB uncompressed across 9 regional files

**8. Wikidata Dump Import (Recommended for Large Imports)** *(v0.9.1)*

For comprehensive imports that avoid SPARQL timeouts, use the Wikidata JSON dump:

```bash
# Download and import from Wikidata dump (~100GB)
# Uses aria2c for fast parallel downloads if available
corp-extractor db import-wikidata-dump --download --limit 50000

# Import only people
corp-extractor db import-wikidata-dump --download --people --no-orgs --limit 100000

# Import only organizations
corp-extractor db import-wikidata-dump --download --orgs --no-people --limit 100000

# Use an existing dump file
corp-extractor db import-wikidata-dump --dump /path/to/latest-all.json.bz2 --limit 50000

# Disable aria2c (use slower single-connection download)
corp-extractor db import-wikidata-dump --download --no-aria2 --limit 10000

# Resume interrupted import (v0.9.2) - loads existing Q codes and skips them
corp-extractor db import-wikidata-dump --dump dump.json.bz2 --resume

# Only import orgs with English Wikipedia articles (stricter filter)
corp-extractor db import-wikidata-dump --download --require-enwiki --orgs --no-people
```

**Fast download with aria2c:**
```bash
# Install aria2c for 10-20x faster downloads (16 parallel connections)
brew install aria2   # macOS
apt install aria2    # Ubuntu/Debian

# Or download manually with more connections:
aria2c -x 32 -s 32 -k 10M \
  https://dumps.wikimedia.org/wikidatawiki/entities/latest-all.json.bz2 \
  -d ~/.cache/corp-extractor \
  -o wikidata-latest-all.json.bz2
```

**Advantages over SPARQL import:**
- No timeouts (processes locally)
- Complete coverage (all notable people/orgs with English Wikipedia articles)
- Captures people like Andy Burnham via occupation (P106) even if position type is generic
- Extracts role dates from position qualifiers (P580/P582)
- Resumable with `--resume` flag (loads existing Q codes and skips them)
- Region normalization using pycountry (handles "UK" → "GB", "California" → "US", etc.)
- Automatic QID label resolution via SPARQL fallback for unresolved references

**Download location:** `~/.cache/corp-extractor/wikidata-latest-all.json.bz2`

### Full Build (Recommended)

For a comprehensive database with all sources:

```bash
# Build complete database (~8M+ records, several hours)
corp-extractor db import-gleif --download
corp-extractor db import-sec --download
corp-extractor db import-companies-house --download
corp-extractor db import-wikidata --limit 100000
corp-extractor db import-people --all --limit 50000  # Notable people

# Canonicalize records (link equivalent entities across sources)
corp-extractor db canonicalize

# Check status
corp-extractor db status

# Create lite version for deployment
corp-extractor db create-lite ~/.cache/corp-extractor/entities.db
```

**Expected output:**
```
Entity Database Status
========================================
Total records: 8,400,000+
Embedding dimension: 768
Database size: ~65 GB (full) / ~20 GB (lite)

Records by source:
  gleif: 3,200,000
  companies_house: 5,000,000
  sec_edgar: 100,000+
  wikidata: 100,000
```

### Database Location

Default: `~/.cache/corp-extractor/entities.db`

Override with `--db` flag:
```bash
corp-extractor db import-gleif --download --db /path/to/entities.db
```

### Organization Canonicalization *(v0.9.2)*

After importing from multiple sources, run canonicalization to link equivalent records:

```bash
corp-extractor db canonicalize
```

**What it does:**
- Finds equivalent organization records across different data sources
- Links them via `canon_id` field pointing to the highest-priority source
- Enables prominence-based search re-ranking

**Matching criteria (in order of priority):**
1. **Same LEI** - GLEIF source_id or Wikidata P1278 (globally unique, no region check)
2. **Same ticker symbol** - globally unique, no region check
3. **Same CIK** - SEC identifier (globally unique, no region check)
4. **Same normalized name + region** - after lowercasing, removing dots
5. **Name with suffix expansion + region** - "Ltd" → "Limited", "Corp" → "Corporation"

**Region normalization** uses pycountry to handle:
- Country codes/names: "GB", "United Kingdom", "Great Britain" → "GB"
- US state codes/names: "CA", "California" → "US"
- Common aliases: "UK" → "GB", "USA" → "US"

**Source priority** (highest to lowest):
1. `gleif` - Gold standard with globally unique LEI
2. `sec_edgar` - Vetted US filers with CIK + ticker
3. `companies_house` - Official UK registry
4. `wikipedia` - Crowdsourced, less authoritative

**Example output:**
```
Canonicalization Results
========================================
Total records processed: 8,400,000
Equivalence groups found: 7,800,000
Multi-record groups: 150,000
Records updated: 8,400,000
```

**Benefits for search:**
- Prominence-based re-ranking boosts companies with records from multiple authoritative sources
- A search for "Microsoft" ranks SEC-filed + GLEIF-registered results higher than Wikipedia-only results
- Canonicalized records get boosts from ALL sources in their group (e.g., +0.08 for ticker, +0.05 for GLEIF, +0.03 for SEC)

## Testing the Database

**Search for an organization:**
```bash
corp-extractor db search "Microsoft"
corp-extractor db search "Barclays" --source companies_house
corp-extractor db search "Apple Inc" --top-k 20 --verbose
```

**Search for a person:** *(v0.9.0)*
```bash
corp-extractor db search-people "Tim Cook"
corp-extractor db search-people "Elon Musk" --top-k 5
corp-extractor db search-people "Warren Buffett" --verbose
```

**Example person search output:**
```
Found 3 results:

  1. Tim Cook (CEO) at Apple Inc [US]
     Source: wikidata:Q312, Type: executive, Score: 0.952

  2. Timothy Donald Cook (CEO) at Apple [US]
     Source: wikidata:Q312, Type: executive, Score: 0.891

  3. Tim Cook (Director) at Nike [US]
     Source: wikidata:Q12345, Type: executive, Score: 0.743
```

**Example output:**
```
Top 5 matches:
1. MICROSOFT CORP
   Source: sec_edgar | ID: 0000789019
   Canonical ID: SEC-CIK:0000789019
   Region: USA
   Similarity: 0.8423

2. Microsoft Corporation
   Source: gleif | ID: INR2EJN1ERAN0W5ZP974
   Canonical ID: LEI:INR2EJN1ERAN0W5ZP974
   Region: US-WA
   Similarity: 0.8156
```

## Publishing to HuggingFace Hub

### Setup

1. Create a HuggingFace account at https://huggingface.co/
2. Generate an access token with write permissions
3. Set the token:
   ```bash
   export HF_TOKEN="hf_..."
   ```

### Upload Database

```bash
# Upload to default repo (Corp-o-Rate-Community/entity-references)
corp-extractor db upload ~/.cache/corp-extractor/entities.db

# Upload to custom repo
corp-extractor db upload entities.db --repo your-org/your-repo

# With custom commit message
corp-extractor db upload entities.db --message "Update with January 2026 GLEIF data"
```

### Download Published Database

Users can download the pre-built database:

```bash
# Download from default repo
corp-extractor db download

# Download from custom repo
corp-extractor db download --repo your-org/your-repo

# Force re-download
corp-extractor db download --force
```

## Using in the Pipeline

The entity database is used by the **EmbeddingCompanyQualifier** plugin in Stage 3 (Qualification). This plugin now returns `CanonicalEntity` objects directly with FQN and qualifiers.

**Python API:**
```python
from statement_extractor.pipeline import ExtractionPipeline

# Database auto-loaded if available
pipeline = ExtractionPipeline()
ctx = pipeline.process("Microsoft acquired Activision Blizzard.")

for stmt in ctx.labeled_statements:
    # FQN format: "LEGAL_NAME (SOURCE,REGION)"
    print(f"{stmt.subject_fqn}")  # e.g., "MICROSOFT CORP (SEC-CIK,USA)"

    # Access qualifiers dict
    subj = stmt.subject_canonical
    if subj.qualifiers_dict:
        print(f"  Legal name: {subj.qualifiers_dict.get('legal_name')}")
        print(f"  Source: {subj.qualifiers_dict.get('source')}")
        print(f"  Region: {subj.qualifiers_dict.get('region')}")
```

**Output format (v0.8.0+):**
```json
{
  "subject": {
    "text": "Microsoft",
    "type": "ORG",
    "name": "MICROSOFT CORP",
    "fqn": "MICROSOFT CORP (SEC-CIK,USA)",
    "canonical_id": "SEC-CIK:0000789019",
    "qualifiers": {
      "legal_name": "MICROSOFT CORP",
      "region": "USA",
      "source": "SEC-CIK",
      "source_id": "0000789019"
    }
  }
}
```

**CLI:**
```bash
# Run pipeline (uses database automatically)
corp-extractor pipeline "Apple announced record earnings."
```

## CLI Reference

| Command | Description |
|---------|-------------|
| `db status` | Show database statistics |
| `db search QUERY` | Search for an organization |
| `db search-people QUERY` | Search for a person *(v0.9.0)* |
| `db import-gleif` | Import GLEIF LEI data |
| `db import-sec` | Import SEC Edgar bulk data |
| `db import-companies-house` | Import UK Companies House data |
| `db import-wikidata` | Import Wikidata organizations (SPARQL) |
| `db import-people` | Import Wikidata notable people (SPARQL) *(v0.9.0)* |
| `db import-sec-officers` | Import SEC Form 4 officers/directors *(v0.9.3)* |
| `db import-ch-officers` | Import Companies House officers (Prod195) *(v0.9.3)* |
| `db import-wikidata-dump` | Import from Wikidata JSON dump (recommended) *(v0.9.1)* |
| `db canonicalize` | Link equivalent records across sources *(v0.9.2)* |
| `db gleif-info` | Show latest GLEIF file info |
| `db download` | Download database from HuggingFace (lite by default) |
| `db download --full` | Download full database with all metadata |
| `db upload` | Upload database to HuggingFace (creates all variants) |
| `db create-lite` | Create lite version (strips record JSON) |
| `db compress` | Compress database with gzip |

### Common Options

| Option | Description |
|--------|-------------|
| `--db PATH` | Database file path |
| `--limit N` | Limit number of records |
| `--batch-size N` | Batch size for commits (default: 50000) |
| `--download` | Download source data automatically |
| `--force` | Force re-download even if cached |
| `-v, --verbose` | Verbose output |

## Database Variants

| Variant | Filename | Contains | Size |
|---------|----------|----------|------|
| **Full** | `entities.db` | All data + embeddings + record metadata | ~65GB |
| **Lite** | `entities-lite.db` | All data + embeddings (no record JSON) | ~20GB |
| **Compressed** | `*.db.gz` | Gzipped versions for transfer | ~50% smaller |

The lite version is recommended for most deployments—it contains everything needed for qualification but omits the full JSON record metadata.

```bash
# Download lite version (default, smaller)
corp-extractor db download

# Download full version with all metadata
corp-extractor db download --full

# Create lite version locally
corp-extractor db create-lite entities.db

# Compress for upload
corp-extractor db compress entities.db
```

## Architecture

```
┌───────────────────────────────────────────────────────────────────────────┐
│                           Entity Database                                  │
├───────────────────────────────────────────────────────────────────────────┤
│  organizations table (SQLite)                                              │
│  ├── id (INTEGER PRIMARY KEY)                                              │
│  ├── name (TEXT)                                                           │
│  ├── name_normalized (TEXT)                                                │
│  ├── region (TEXT)                                                         │
│  ├── entity_type (TEXT)                                                    │
│  ├── source (TEXT: gleif|sec_edgar|companies_house|wikidata)               │
│  ├── source_id (TEXT)                                                      │
│  ├── from_date (TEXT) - founding/registration date (ISO format)            │
│  ├── to_date (TEXT) - dissolution date (ISO format)                        │
│  ├── canon_id (INTEGER) - ID of canonical record (v0.9.2)                  │
│  ├── canon_size (INTEGER) - size of canonical group (v0.9.2)               │
│  └── record (JSON) - omitted in lite version                               │
│  UNIQUE(source, source_id)                                                 │
├───────────────────────────────────────────────────────────────────────────┤
│  organization_embeddings (sqlite-vec virtual table)                        │
│  ├── org_id (INTEGER)                                                      │
│  └── embedding (FLOAT[768])                                                │
├───────────────────────────────────────────────────────────────────────────┤
│  people table (SQLite) - v0.9.3                                            │
│  ├── id (INTEGER PRIMARY KEY)                                              │
│  ├── name (TEXT)                                                           │
│  ├── name_normalized (TEXT)                                                │
│  ├── source (TEXT: wikidata|sec_edgar|companies_house)                     │
│  ├── source_id (TEXT)                                                      │
│  ├── country (TEXT)                                                        │
│  ├── person_type (TEXT: executive|politician|government|military|...)      │
│  ├── known_for_role (TEXT) - e.g., "CEO", "President"                      │
│  ├── known_for_org (TEXT) - e.g., "Apple Inc", "Tesla"                     │
│  ├── known_for_org_id (INTEGER FK) - references organizations(id)          │
│  ├── from_date (TEXT) - role start date (ISO format)                       │
│  ├── to_date (TEXT) - role end date (ISO format)                           │
│  ├── birth_date (TEXT) - date of birth (ISO format) (v0.9.2)               │
│  ├── death_date (TEXT) - date of death (ISO format) (v0.9.2)               │
│  └── record (JSON) - omitted in lite version                               │
│  UNIQUE(source, source_id, known_for_role, known_for_org)                  │
├───────────────────────────────────────────────────────────────────────────┤
│  person_embeddings (sqlite-vec virtual table)                              │
│  ├── person_id (INTEGER)                                                   │
│  └── embedding (FLOAT[768]) - combines name|role|org                       │
├───────────────────────────────────────────────────────────────────────────┤
│  qid_labels table (SQLite) - v0.9.2                                        │
│  ├── qid (TEXT PRIMARY KEY) - Wikidata QID (e.g., "Q30")                   │
│  └── label (TEXT) - resolved label (e.g., "United States")                 │
│  Used for caching QID → label mappings during import                       │
└───────────────────────────────────────────────────────────────────────────┘
         │
         │ Vector similarity search
         ▼
┌───────────────────────────────────────────────────────────────────────────┐
│  Stage 3 Qualifier Plugins                                                 │
├───────────────────────────────────────────────────────────────────────────┤
│  EmbeddingCompanyQualifier (ORG entities)                                  │
│  ├── Search by organization name embedding                                 │
│  ├── Return CanonicalEntity with FQN and qualifiers                        │
│  └── Optional LLM confirmation for ambiguous matches                       │
├───────────────────────────────────────────────────────────────────────────┤
│  PersonQualifier (PERSON entities) - v0.9.0                                │
│  ├── LLM extracts role + org from context                                  │
│  ├── Search PersonDatabase with role/org boost                             │
│  ├── Resolve org mentions via OrganizationResolver                         │
│  └── Build CanonicalEntity with Wikidata ID + resolved role/org            │
└───────────────────────────────────────────────────────────────────────────┘
```

## Technical Details

### How It Works

1. **Embedding Generation**: Company names are converted to 768-dimensional vectors using Google's `embeddinggemma-300m` model (via sentence-transformers). This model captures semantic meaning, so "Microsoft Corp" and "Microsoft Corporation" produce similar embeddings.

2. **Vector Storage**: Embeddings are stored in a SQLite database using the `sqlite-vec` extension, which provides efficient approximate nearest neighbor (ANN) search.

3. **Similarity Search**: When searching for a company, the query is embedded and compared against all stored embeddings using cosine similarity. The top-K most similar records are returned.

4. **Canonical ID Resolution**: Each match includes a canonical identifier in the format `SOURCE-PREFIX:source_id` (e.g., `LEI:INR2EJN1ERAN0W5ZP974`, `SEC-CIK:0000789019`), which can be used for deduplication and cross-referencing.

### Embedding Model

| Property | Value |
|----------|-------|
| Model | `google/embeddinggemma-300m` |
| Dimensions | 768 |
| Framework | sentence-transformers |
| Size | ~300M parameters |
| Device | Auto-detected (CUDA, MPS, or CPU) |

The model is downloaded automatically on first use (~600MB).

### Vector Search Performance

| Database Size | Search Time | Memory |
|---------------|-------------|--------|
| 100K records | ~50ms | ~500MB |
| 1M records | ~200ms | ~3GB |
| 8M records | ~500ms | ~20GB |

Search is performed using sqlite-vec's virtual table, which uses exact cosine similarity (not approximate). For very large databases, consider filtering by source.

### Similarity Thresholds

| Score | Interpretation |
|-------|----------------|
| > 0.85 | Strong match (likely same entity) |
| 0.70 - 0.85 | Good match (probable same entity) |
| 0.55 - 0.70 | Moderate match (may need verification) |
| < 0.55 | Weak match (likely different entity) |

### Record Schema

Each organization record contains:

```python
class CompanyRecord(BaseModel):
    name: str           # Searchable name (used for embedding)
    region: str         # Geographic region (e.g., "USA", "GB", "US-CA")
    entity_type: str    # Organization type (business, nonprofit, government, etc.)
    source: str         # Data source identifier
    source_id: str      # Unique ID from source
    canonical_id: str   # Full canonical ID (e.g., "SEC-CIK:0000789019")
    from_date: str      # Founding/registration date (ISO format YYYY-MM-DD)
    to_date: str        # Dissolution date (ISO format YYYY-MM-DD)
    record: dict        # Full record from source (omitted in lite version)
```

Each person record contains *(v0.9.2)*:

```python
class PersonRecord(BaseModel):
    name: str               # Display name (used for embedding and display)
    source: str             # Data source (wikidata, sec_edgar, companies_house)
    source_id: str          # Wikidata QID (e.g., "Q312")
    country: str            # Country code or name (e.g., "US", "Germany")
    person_type: PersonType # Classification (executive, politician, etc.)
    known_for_role: str     # Primary role from Wikipedia (e.g., "CEO")
    known_for_org: str      # Primary org from Wikipedia (e.g., "Apple Inc")
    from_date: str          # Role start date (ISO format YYYY-MM-DD)
    to_date: str            # Role end date (ISO format YYYY-MM-DD)
    birth_date: str         # Date of birth (ISO format YYYY-MM-DD)
    death_date: str         # Date of death (ISO format YYYY-MM-DD)
    record: dict            # Full record from source (omitted in lite version)

    @property
    def is_historic(self) -> bool:
        """Return True if the person is deceased."""
        return bool(self.death_date)

class PersonType(Enum):
    EXECUTIVE = "executive"      # CEOs, board members, C-suite
    POLITICIAN = "politician"    # Elected officials (presidents, MPs, mayors)
    GOVERNMENT = "government"    # Civil servants, diplomats, appointed officials
    MILITARY = "military"        # Military officers, armed forces personnel
    LEGAL = "legal"              # Judges, lawyers, legal professionals
    PROFESSIONAL = "professional"# Known for profession (doctors, engineers)
    ACADEMIC = "academic"        # Professors, researchers
    ARTIST = "artist"            # Traditional creatives (musicians, actors, painters)
    MEDIA = "media"              # Internet/social media personalities (YouTubers, influencers)
    ATHLETE = "athlete"          # Sports figures
    ENTREPRENEUR = "entrepreneur"# Founders, business owners
    JOURNALIST = "journalist"    # Reporters, news presenters, columnists
    ACTIVIST = "activist"        # Advocates, campaigners
    SCIENTIST = "scientist"      # Scientists, inventors
    UNKNOWN = "unknown"          # Type not determined
```

**Note on person records:** The same person (same `source_id`) can have multiple records with different `known_for_role` and `known_for_org` combinations. For example, Tim Cook may have records for both "CEO at Apple Inc" and "Board Director at Nike".

**Source identifier prefixes:**

| Source | Prefix | Example Canonical ID |
|--------|--------|---------------------|
| GLEIF | `LEI` | `LEI:INR2EJN1ERAN0W5ZP974` |
| SEC Edgar | `SEC-CIK` | `SEC-CIK:0000789019` |
| Companies House | `UK-CH` | `UK-CH:00445790` |
| Wikidata (org) | `WIKIDATA` | `WIKIDATA:Q2283` |
| Wikidata (person) | `wikidata` | `wikidata:Q312` |

**Source-specific record fields:**

| Source | Key Fields |
|--------|------------|
| GLEIF | `lei`, `jurisdiction`, `country`, `city`, `status`, `other_names`, `initial_registration_date` |
| SEC Edgar | `cik`, `ticker`, `title`, `first_filing_date` |
| Companies House | `company_number`, `company_status`, `company_type`, `date_of_creation`, `date_of_cessation`, `sic_code` |
| Wikidata (orgs) | `wikidata_id`, `label`, `lei`, `ticker`, `exchange`, `country`, `inception`, `dissolution` |
| Wikidata (people) | `wikidata_id`, `label`, `role`, `org`, `country`, `from_date`, `to_date` |
| SEC Form 4 (people) | `owner_cik`, `issuer_cik`, `issuer_name`, `issuer_ticker`, `is_director`, `is_officer`, `officer_title` |
| Companies House (people) | `person_id`, `company_number`, `company_name`, `occupation`, `nationality`, `postcode`, `is_current` |

### Import Pipeline

```
Source Data → Parser → CompanyRecord → Embedder → Database
     │           │           │            │           │
     │           │           │            │           └── sqlite-vec INSERT
     │           │           │            └── embeddinggemma-300m
     │           │           └── Pydantic validation
     │           └── Source-specific (XML, JSON, CSV)
     └── Download (GLEIF ZIP, SEC JSON, CH CSV, Wikidata SPARQL)
```

**Batch processing:**
- Records are processed in batches (default: 50,000)
- Each batch: embed names → insert records → insert embeddings
- Progress logged every 10,000 records

### Caching Strategy

Downloaded source files are cached to avoid re-downloading:

| Source | Cache Location | Cache Key |
|--------|----------------|-----------|
| GLEIF | `/tmp/gleif/` | `file_id` from API |
| Companies House | `/tmp/companies_house/` | `file_date` from page |
| SEC Edgar | (not cached) | Always fresh from API |
| Wikidata | (not cached) | Live SPARQL queries |

Use `--force` to re-download cached files.

### Deduplication

The database does not automatically deduplicate across sources. The same company may appear multiple times:
- Microsoft in GLEIF → `LEI:INR2EJN1ERAN0W5ZP974`
- Microsoft in SEC Edgar → `SEC-CIK:0000789019`
- Microsoft in Wikidata → `WIKIDATA:Q2283`

This is intentional—different sources provide different identifiers and metadata. The qualifier plugin returns the first confident match based on embedding similarity.

## Data Freshness

| Source | Update Frequency | Notes |
|--------|------------------|-------|
| GLEIF | Daily | New file published each day |
| SEC Edgar | Real-time | API always current |
| Companies House | Monthly | Bulk file updated monthly |
| Wikidata | Continuous | SPARQL returns live data |

To update the database:
```bash
# Re-download and reimport with --force
corp-extractor db import-gleif --download --force
corp-extractor db import-companies-house --download --force
```

## Troubleshooting

**"sqlite-vec extension not found"**
```bash
pip install sqlite-vec
```

**GLEIF download timeout**
The GLEIF file is ~1.5GB. Ensure stable internet connection. The file is cached after first download.

**Wikidata SPARQL timeout**
Wikidata's endpoint can be slow. Use `--limit` to reduce query size:
```bash
corp-extractor db import-wikidata --limit 10000
```

For large imports that avoid timeouts entirely, use the dump-based importer:
```bash
corp-extractor db import-wikidata-dump --download --limit 50000
```

**Companies House API 401 Unauthorized**
Ensure you have a valid API key from the **Companies House API** (not Streaming API):
https://developer.company-information.service.gov.uk/

**Database too large**
For smaller deployments, import only needed sources:
```bash
# Just SEC (US public companies)
corp-extractor db import-sec

# Just UK companies
corp-extractor db import-companies-house --download --limit 100000
```

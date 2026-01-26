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
| `entities-lite.db` | ~50GB | Core fields + embeddings only |
| `entities.db` | ~74GB | Full records with source metadata |

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

```
+---------------------------------------------------------------------------+
|                           Entity Database                                  |
+---------------------------------------------------------------------------+
|  organizations table (SQLite)                                              |
|  +-- id (INTEGER PRIMARY KEY)                                              |
|  +-- name (TEXT)                                                           |
|  +-- name_normalized (TEXT)                                                |
|  +-- region (TEXT)                                                         |
|  +-- entity_type (TEXT)                                                    |
|  +-- source (TEXT: gleif|sec_edgar|companies_house|wikidata)               |
|  +-- source_id (TEXT)                                                      |
|  +-- from_date (TEXT) - founding/registration date (ISO format)            |
|  +-- to_date (TEXT) - dissolution date (ISO format)                        |
|  +-- canon_id (INTEGER) - ID of canonical record (v0.9.2)                  |
|  +-- canon_size (INTEGER) - size of canonical group (v0.9.2)               |
|  +-- record (JSON) - omitted in lite version                               |
|  UNIQUE(source, source_id)                                                 |
+---------------------------------------------------------------------------+
|  organization_embeddings (sqlite-vec virtual table)                        |
|  +-- org_id (INTEGER)                                                      |
|  +-- embedding (FLOAT[768])                                                |
+---------------------------------------------------------------------------+
|  people table (SQLite) - v0.9.3                                            |
|  +-- id (INTEGER PRIMARY KEY)                                              |
|  +-- name (TEXT)                                                           |
|  +-- name_normalized (TEXT)                                                |
|  +-- source (TEXT: wikidata|sec_edgar|companies_house)                     |
|  +-- source_id (TEXT)                                                      |
|  +-- country (TEXT)                                                        |
|  +-- person_type (TEXT: executive|politician|government|military|...)      |
|  +-- known_for_role (TEXT) - e.g., "CEO", "President"                      |
|  +-- known_for_org (TEXT) - e.g., "Apple Inc", "Tesla"                     |
|  +-- known_for_org_id (INTEGER FK) - references organizations(id)          |
|  +-- from_date (TEXT) - role start date (ISO format)                       |
|  +-- to_date (TEXT) - role end date (ISO format)                           |
|  +-- birth_date (TEXT) - date of birth (ISO format) (v0.9.2)               |
|  +-- death_date (TEXT) - date of death (ISO format) (v0.9.2)               |
|  +-- record (JSON) - omitted in lite version                               |
|  UNIQUE(source, source_id, known_for_role, known_for_org)                  |
+---------------------------------------------------------------------------+
|  person_embeddings (sqlite-vec virtual table)                              |
|  +-- person_id (INTEGER)                                                   |
|  +-- embedding (FLOAT[768]) - combines name|role|org                       |
+---------------------------------------------------------------------------+
|  qid_labels table (SQLite) - v0.9.2                                        |
|  +-- qid (TEXT PRIMARY KEY) - Wikidata QID (e.g., "Q30")                   |
|  +-- label (TEXT) - resolved label (e.g., "United States")                 |
|  Used for caching QID -> label mappings during import                      |
+---------------------------------------------------------------------------+
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

## Data Sources

### Organizations

| Source | Records | Identifier | Date Fields | Coverage |
|--------|---------|------------|-------------|----------|
| [GLEIF](https://www.gleif.org/) | ~3.2M | LEI (Legal Entity Identifier) | `from_date`: LEI registration date | Global companies with LEI |
| [SEC Edgar](https://www.sec.gov/) | ~100K+ | CIK (Central Index Key) | `from_date`: First SEC filing date | All SEC filers (not just public companies) |
| [Companies House](https://www.gov.uk/government/organisations/companies-house) | ~5M | Company Number | `from_date`: Incorporation, `to_date`: Dissolution | UK registered companies |
| [Wikidata](https://www.wikidata.org/) | Variable | QID | `from_date`: Inception (P571), `to_date`: Dissolution (P576) | Notable companies worldwide |

### People

| Source | Records | Identifier | Date Fields | Coverage |
|--------|---------|------------|-------------|----------|
| [Wikidata](https://www.wikidata.org/) | Variable | QID | `from_date`: Position start (P580), `to_date`: Position end (P582) | Notable people with English Wikipedia articles |
| [SEC Form 4](https://www.sec.gov/) *(v0.9.3)* | ~280K/year | Owner CIK | `from_date`: Period of report | US public company officers, directors, 10%+ owners |
| [Companies House](https://www.gov.uk/government/organisations/companies-house) *(v0.9.3)* | ~15M+ | Person number | `from_date`: Appointment date, `to_date`: Resignation date | UK company officers (directors, secretaries) |

### Embedding Model

| Property | Value |
|----------|-------|
| Model | `google/embeddinggemma-300m` |
| Dimensions | 768 |
| Framework | sentence-transformers |
| Size | ~300M parameters |
| Device | Auto-detected (CUDA, MPS, or CPU) |

The model is downloaded automatically on first use (~600MB).

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
- Region normalization using pycountry (handles "UK" -> "GB", "California" -> "US", etc.)
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

## Canonicalization

After importing from multiple sources, run canonicalization to link equivalent records:

```bash
corp-extractor db canonicalize
```

### Organization Canonicalization *(v0.9.2)*

**Matching criteria (in order of priority):**
1. **Same LEI** - GLEIF source_id or Wikidata P1278 (globally unique, no region check)
2. **Same ticker symbol** - globally unique, no region check
3. **Same CIK** - SEC identifier (globally unique, no region check)
4. **Same normalized name + region** - after lowercasing, removing dots
5. **Name with suffix expansion + region** - "Ltd" -> "Limited", "Corp" -> "Corporation"

**Region normalization** uses pycountry to handle:
- Country codes/names: "GB", "United Kingdom", "Great Britain" -> "GB"
- US state codes/names: "CA", "California" -> "US"
- Common aliases: "UK" -> "GB", "USA" -> "US"

**Source priority** (highest to lowest):
1. `gleif` - Gold standard with globally unique LEI
2. `sec_edgar` - Vetted US filers with CIK + ticker
3. `companies_house` - Official UK registry
4. `wikipedia` - Crowdsourced, less authoritative

### People Canonicalization *(v0.9.3)*

**Matching criteria:**
- Records matched by normalized name + same organization (using org canonical group)
- Records matched by normalized name + overlapping date ranges

**Source priority** (highest to lowest):
1. `wikidata` - Most comprehensive notable people
2. `sec_edgar` - Verified from SEC filings
3. `companies_house` - UK company officers

**Benefits for search:**
- Prominence-based re-ranking boosts entities with records from multiple authoritative sources
- A search for "Microsoft" ranks SEC-filed + GLEIF-registered results higher than Wikipedia-only results
- Canonicalized records get boosts from ALL sources in their group (e.g., +0.08 for ticker, +0.05 for GLEIF, +0.03 for SEC)

## CLI Reference

### Core Commands

| Command | Description |
|---------|-------------|
| `db status` | Show database statistics |
| `db search QUERY` | Search for an organization |
| `db search-people QUERY` | Search for a person *(v0.9.0)* |
| `db canonicalize` | Link equivalent records across sources *(v0.9.2)* |

### Import Commands

| Command | Description |
|---------|-------------|
| `db import-gleif` | Import GLEIF LEI data |
| `db import-sec` | Import SEC Edgar bulk data |
| `db import-companies-house` | Import UK Companies House data |
| `db import-wikidata` | Import Wikidata organizations (SPARQL) |
| `db import-people` | Import Wikidata notable people (SPARQL) *(v0.9.0)* |
| `db import-sec-officers` | Import SEC Form 4 officers/directors *(v0.9.3)* |
| `db import-ch-officers` | Import Companies House officers (Prod195) *(v0.9.3)* |
| `db import-wikidata-dump` | Import from Wikidata JSON dump (recommended) *(v0.9.1)* |
| `db gleif-info` | Show latest GLEIF file info |

### Hub Commands

| Command | Description |
|---------|-------------|
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

### Search Examples

```bash
# Search organizations
corp-extractor db search "Microsoft"
corp-extractor db search "Barclays" --source companies_house
corp-extractor db search "Apple Inc" --top-k 20 --verbose

# Search people
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

**Example organization search output:**
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
    source: str                  # 'wikidata', 'sec_edgar', 'companies_house'
    source_id: str               # Wikidata QID, Owner CIK, or Person number
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

### PersonType Enum

```python
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

### Source Identifier Prefixes

| Source | Prefix | Example Canonical ID |
|--------|--------|---------------------|
| GLEIF | `LEI` | `LEI:INR2EJN1ERAN0W5ZP974` |
| SEC Edgar | `SEC-CIK` | `SEC-CIK:0000789019` |
| Companies House | `UK-CH` | `UK-CH:00445790` |
| Wikidata (org) | `WIKIDATA` | `WIKIDATA:Q2283` |
| Wikidata (person) | `wikidata` | `wikidata:Q312` |

### Source-specific Record Fields

| Source | Key Fields |
|--------|------------|
| GLEIF | `lei`, `jurisdiction`, `country`, `city`, `status`, `other_names`, `initial_registration_date` |
| SEC Edgar | `cik`, `ticker`, `title`, `first_filing_date` |
| Companies House | `company_number`, `company_status`, `company_type`, `date_of_creation`, `date_of_cessation`, `sic_code` |
| Wikidata (orgs) | `wikidata_id`, `label`, `lei`, `ticker`, `exchange`, `country`, `inception`, `dissolution` |
| Wikidata (people) | `wikidata_id`, `label`, `role`, `org`, `country`, `from_date`, `to_date` |
| SEC Form 4 (people) | `owner_cik`, `issuer_cik`, `issuer_name`, `issuer_ticker`, `is_director`, `is_officer`, `officer_title` |
| Companies House (people) | `person_id`, `company_number`, `company_name`, `occupation`, `nationality`, `postcode`, `is_current` |

## Technical Details

### How It Works

1. **Embedding Generation**: Entity names are converted to 768-dimensional vectors using Google's `embeddinggemma-300m` model (via sentence-transformers). This model captures semantic meaning, so "Microsoft Corp" and "Microsoft Corporation" produce similar embeddings.

2. **Vector Storage**: Embeddings are stored in a SQLite database using the `sqlite-vec` extension, which provides efficient approximate nearest neighbor (ANN) search.

3. **Similarity Search**: When searching for an entity, the query is embedded and compared against all stored embeddings using cosine similarity. The top-K most similar records are returned.

4. **Canonical ID Resolution**: Each match includes a canonical identifier in the format `SOURCE-PREFIX:source_id` (e.g., `LEI:INR2EJN1ERAN0W5ZP974`, `SEC-CIK:0000789019`), which can be used for deduplication and cross-referencing.

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

### Import Pipeline

```
Source Data -> Parser -> Record -> Embedder -> Database
     |           |          |         |           |
     |           |          |         |           +-- sqlite-vec INSERT
     |           |          |         +-- embeddinggemma-300m
     |           |          +-- Pydantic validation
     |           +-- Source-specific (XML, JSON, CSV)
     +-- Download (GLEIF ZIP, SEC JSON, CH CSV, Wikidata SPARQL)
```

**Batch processing:**
- Records are processed in batches (default: 50,000)
- Each batch: embed names -> insert records -> insert embeddings
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
- Microsoft in GLEIF -> `LEI:INR2EJN1ERAN0W5ZP974`
- Microsoft in SEC Edgar -> `SEC-CIK:0000789019`
- Microsoft in Wikidata -> `WIKIDATA:Q2283`

This is intentional - different sources provide different identifiers and metadata. The qualifier plugin returns the first confident match based on embedding similarity. Run `db canonicalize` to link equivalent records.

### Data Freshness

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

### GLEIF Download Timeout

The GLEIF file is ~1.5GB. Ensure stable internet connection. The file is cached after first download.

### Wikidata SPARQL Timeout

Wikidata's endpoint can be slow. Use `--limit` to reduce query size:
```bash
corp-extractor db import-wikidata --limit 10000
```

For large imports that avoid timeouts entirely, use the dump-based importer:
```bash
corp-extractor db import-wikidata-dump --download --limit 50000
```

### Companies House API 401 Unauthorized

Ensure you have a valid API key from the **Companies House API** (not Streaming API):
https://developer.company-information.service.gov.uk/

### Database Too Large

For smaller deployments, import only needed sources:
```bash
# Just SEC (US public companies)
corp-extractor db import-sec

# Just UK companies
corp-extractor db import-companies-house --download --limit 100000
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
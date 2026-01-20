# Company Embedding Database

The company embedding database enables fast entity qualification by matching company names against pre-computed embeddings from authoritative sources.

## Overview

The database uses `sqlite-vec` for vector similarity search, storing company records with:
- **name**: Searchable company name
- **embedding**: 768-dimensional vector from `google/embeddinggemma-300m`
- **legal_name**: Official registered name
- **source**: Data source (gleif, sec_edgar, companies_house, wikipedia)
- **source_id**: Unique identifier (LEI, CIK, company number, Wikidata QID)
- **record**: Full JSON record from source

## Data Sources

| Source | Records | Identifier | Coverage |
|--------|---------|------------|----------|
| [GLEIF](https://www.gleif.org/) | ~3.2M | LEI (Legal Entity Identifier) | Global companies with LEI |
| [SEC Edgar](https://www.sec.gov/) | ~10K | CIK (Central Index Key) | US public companies |
| [Companies House](https://www.gov.uk/government/organisations/companies-house) | ~5M | Company Number | UK registered companies |
| [Wikidata](https://www.wikidata.org/) | Variable | QID | Notable companies worldwide |

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

**2. SEC Edgar (US Public Companies)**

```bash
# Import all SEC companies (~10K records)
corp-extractor db import-sec

# Import with limit
corp-extractor db import-sec --limit 5000
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

**4. Wikidata (Notable Companies)**

```bash
# Import via SPARQL query
corp-extractor db import-wikidata --limit 50000

# Import all companies (slower, may timeout)
corp-extractor db import-wikidata --all
```

### Full Build (Recommended)

For a comprehensive database with all sources:

```bash
# Build complete database (~8M+ records, several hours)
corp-extractor db import-gleif --download
corp-extractor db import-sec
corp-extractor db import-companies-house --download
corp-extractor db import-wikidata --limit 100000

# Check status
corp-extractor db status
```

**Expected output:**
```
Company Database Status
========================================
Total records: 8,300,000+
Embedding dimension: 768
Database size: ~65 GB

Records by source:
  gleif: 3,200,000
  companies_house: 5,000,000
  sec_edgar: 10,000
  wikipedia: 100,000
```

### Database Location

Default: `~/.cache/corp-extractor/companies.db`

Override with `--db` flag:
```bash
corp-extractor db import-gleif --download --db /path/to/companies.db
```

## Testing the Database

**Search for a company:**
```bash
corp-extractor db search "Microsoft"
corp-extractor db search "Barclays" --source companies_house
corp-extractor db search "Apple Inc" --top-k 20 --verbose
```

**Example output:**
```
Top 5 matches:
1. MICROSOFT CORP
   Source: sec_edgar | ID: 0000789019
   Canonical ID: sec_edgar:0000789019
   Similarity: 0.8423

2. Microsoft Corporation
   Source: gleif | ID: INR2EJN1ERAN0W5ZP974
   Canonical ID: gleif:INR2EJN1ERAN0W5ZP974
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
# Upload to default repo (Corp-o-Rate-Community/company-embeddings)
corp-extractor db upload ~/.cache/corp-extractor/companies.db

# Upload to custom repo
corp-extractor db upload companies.db --repo your-org/your-repo

# With custom commit message
corp-extractor db upload companies.db --message "Update with January 2026 GLEIF data"
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

The company database is used by the **EmbeddingCompanyQualifier** plugin in Stage 3 (Qualification).

**Python API:**
```python
from statement_extractor.pipeline import ExtractionPipeline

# Database auto-loaded if available
pipeline = ExtractionPipeline()
ctx = pipeline.process("Microsoft acquired Activision Blizzard.")

for stmt in ctx.labeled_statements:
    print(f"{stmt.subject_fqn}")  # e.g., "Microsoft (sec_edgar:0000789019)"
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
| `db search QUERY` | Search for a company |
| `db import-gleif` | Import GLEIF LEI data |
| `db import-sec` | Import SEC Edgar data |
| `db import-companies-house` | Import UK Companies House data |
| `db import-wikidata` | Import Wikidata companies |
| `db gleif-info` | Show latest GLEIF file info |
| `db download` | Download database from HuggingFace |
| `db upload` | Upload database to HuggingFace |

### Common Options

| Option | Description |
|--------|-------------|
| `--db PATH` | Database file path |
| `--limit N` | Limit number of records |
| `--batch-size N` | Batch size for commits (default: 50000) |
| `--download` | Download source data automatically |
| `--force` | Force re-download even if cached |
| `-v, --verbose` | Verbose output |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Company Database                         │
├─────────────────────────────────────────────────────────────┤
│  companies table (SQLite)                                   │
│  ├── id (INTEGER PRIMARY KEY)                               │
│  ├── name (TEXT)                                            │
│  ├── embedding_name (TEXT)                                  │
│  ├── legal_name (TEXT)                                      │
│  ├── source (TEXT: gleif|sec_edgar|companies_house|wikipedia)│
│  ├── source_id (TEXT)                                       │
│  └── record (JSON)                                          │
├─────────────────────────────────────────────────────────────┤
│  company_embeddings (sqlite-vec virtual table)              │
│  ├── id (INTEGER)                                           │
│  └── embedding (FLOAT[768])                                 │
└─────────────────────────────────────────────────────────────┘
         │
         │ Vector similarity search
         ▼
┌─────────────────────────────────────────────────────────────┐
│  EmbeddingCompanyQualifier (Stage 3 Plugin)                 │
│  ├── Search by company name embedding                       │
│  ├── Return top-K candidates with similarity scores         │
│  └── Optional LLM confirmation for ambiguous matches        │
└─────────────────────────────────────────────────────────────┘
```

## Technical Details

### How It Works

1. **Embedding Generation**: Company names are converted to 768-dimensional vectors using Google's `embeddinggemma-300m` model (via sentence-transformers). This model captures semantic meaning, so "Microsoft Corp" and "Microsoft Corporation" produce similar embeddings.

2. **Vector Storage**: Embeddings are stored in a SQLite database using the `sqlite-vec` extension, which provides efficient approximate nearest neighbor (ANN) search.

3. **Similarity Search**: When searching for a company, the query is embedded and compared against all stored embeddings using cosine similarity. The top-K most similar records are returned.

4. **Canonical ID Resolution**: Each match includes a canonical identifier in the format `source:source_id` (e.g., `gleif:INR2EJN1ERAN0W5ZP974`), which can be used for deduplication and cross-referencing.

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

Each company record contains:

```python
class CompanyRecord(BaseModel):
    name: str           # Searchable name (used for embedding)
    embedding_name: str # Name used for embedding generation
    legal_name: str     # Official registered name
    source: str         # Data source identifier
    source_id: str      # Unique ID from source
    record: dict        # Full record from source (varies by source)
```

**Source-specific record fields:**

| Source | Key Fields |
|--------|------------|
| GLEIF | `lei`, `jurisdiction`, `country`, `city`, `status`, `other_names` |
| SEC Edgar | `cik`, `ticker`, `title` |
| Companies House | `company_number`, `company_status`, `company_type`, `date_of_creation`, `sic_code` |
| Wikidata | `wikidata_id`, `label`, `lei`, `ticker`, `exchange`, `country` |

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
- Microsoft in GLEIF (LEI: INR2EJN1ERAN0W5ZP974)
- Microsoft in SEC Edgar (CIK: 0000789019)
- Microsoft in Wikidata (QID: Q2283)

This is intentional—different sources provide different identifiers and metadata. The pipeline can use multiple matches to enrich entity qualification.

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

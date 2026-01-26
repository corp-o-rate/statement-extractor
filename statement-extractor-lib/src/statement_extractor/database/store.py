"""
Entity/Organization database with sqlite-vec for vector search.

Uses a hybrid approach:
1. Text-based filtering to narrow candidates (Levenshtein-like)
2. sqlite-vec vector search for semantic ranking
"""

import json
import logging
import re
import sqlite3
import time
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import pycountry
import sqlite_vec

from .models import CompanyRecord, DatabaseStats, EntityType, PersonRecord, PersonType

logger = logging.getLogger(__name__)

# Default database location
DEFAULT_DB_PATH = Path.home() / ".cache" / "corp-extractor" / "entities.db"

# Module-level shared connections by path (both databases share the same connection)
_shared_connections: dict[str, sqlite3.Connection] = {}

# Module-level singleton for OrganizationDatabase to prevent multiple loads
_database_instances: dict[str, "OrganizationDatabase"] = {}

# Module-level singleton for PersonDatabase
_person_database_instances: dict[str, "PersonDatabase"] = {}


def _get_shared_connection(db_path: Path, embedding_dim: int = 768) -> sqlite3.Connection:
    """Get or create a shared database connection for the given path."""
    path_key = str(db_path)
    if path_key not in _shared_connections:
        # Ensure directory exists
        db_path.parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row

        # Load sqlite-vec extension
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)

        _shared_connections[path_key] = conn
        logger.debug(f"Created shared database connection for {path_key}")

    return _shared_connections[path_key]


def close_shared_connection(db_path: Optional[Path] = None) -> None:
    """Close a shared database connection."""
    path_key = str(db_path or DEFAULT_DB_PATH)
    if path_key in _shared_connections:
        _shared_connections[path_key].close()
        del _shared_connections[path_key]
        logger.debug(f"Closed shared database connection for {path_key}")

# Comprehensive set of corporate legal suffixes (international)
COMPANY_SUFFIXES: set[str] = {
    'A/S', 'AB', 'AG', 'AO', 'AG & Co', 'AG &', 'AG & CO.', 'AG & CO. KG', 'AG & CO. KGaA',
    'AG & KG', 'AG & KGaA', 'AG & PARTNER', 'ATE', 'ASA', 'B.V.', 'BV', 'Class A', 'Class B',
    'Class C', 'Class D', 'Class E', 'Class F', 'Class G', 'CO', 'Co', 'Co.', 'Company',
    'Corp', 'Corp.', 'Corporation', 'DAC', 'GmbH', 'Inc', 'Inc.', 'Incorporated', 'KGaA',
    'Limited', 'LLC', 'LLP', 'LP', 'Ltd', 'Ltd.', 'N.V.', 'NV', 'Plc', 'PC', 'plc', 'PLC',
    'Pty Ltd', 'Pty', 'Pty. Ltd.', 'S.A.', 'S.A.B. de C.V.', 'SAB de CV', 'S.A.B.', 'S.A.P.I.',
    'NV/SA', 'SDI', 'SpA', 'S.L.', 'S.p.A.', 'SA', 'SE', 'Tbk PT', 'U.A.',
    # Additional common suffixes
    'Group', 'Holdings', 'Holding', 'Partners', 'Trust', 'Fund', 'Bank', 'N.A.', 'The',
}

# Source priority for organization canonicalization (lower = higher priority)
SOURCE_PRIORITY: dict[str, int] = {
    "gleif": 1,       # Gold standard LEI - globally unique legal entity identifier
    "sec_edgar": 2,   # Vetted US filers with CIK + ticker
    "companies_house": 3,  # Official UK registry
    "wikipedia": 4,   # Crowdsourced, less authoritative
}

# Source priority for people canonicalization (lower = higher priority)
PERSON_SOURCE_PRIORITY: dict[str, int] = {
    "wikidata": 1,       # Curated, has rich biographical data and Q codes
    "sec_edgar": 2,      # Vetted US filers (Form 4 officers/directors)
    "companies_house": 3,  # UK company officers
}

# Suffix expansions for canonical name matching
SUFFIX_EXPANSIONS: dict[str, str] = {
    " ltd": " limited",
    " corp": " corporation",
    " inc": " incorporated",
    " co": " company",
    " intl": " international",
    " natl": " national",
}


class UnionFind:
    """Simple Union-Find (Disjoint Set Union) data structure for canonicalization."""

    def __init__(self, elements: list[int]):
        """Initialize with list of element IDs."""
        self.parent: dict[int, int] = {e: e for e in elements}
        self.rank: dict[int, int] = {e: 0 for e in elements}

    def find(self, x: int) -> int:
        """Find with path compression."""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int) -> None:
        """Union by rank."""
        px, py = self.find(x), self.find(y)
        if px == py:
            return
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1

    def groups(self) -> dict[int, list[int]]:
        """Return dict of root -> list of members."""
        result: dict[int, list[int]] = {}
        for e in self.parent:
            root = self.find(e)
            result.setdefault(root, []).append(e)
        return result


# Common region aliases not handled well by pycountry fuzzy search
REGION_ALIASES: dict[str, str] = {
    "uk": "GB",
    "u.k.": "GB",
    "england": "GB",
    "scotland": "GB",
    "wales": "GB",
    "northern ireland": "GB",
    "usa": "US",
    "u.s.a.": "US",
    "u.s.": "US",
    "united states of america": "US",
    "america": "US",
}

# Cache for region normalization lookups
_region_cache: dict[str, str] = {}


def _normalize_region(region: str) -> str:
    """
    Normalize a region string to ISO 3166-1 alpha-2 country code.

    Handles:
    - Country codes (2-letter, 3-letter)
    - Country names (with fuzzy matching)
    - US state codes (CA, NY) -> US
    - US state names (California, New York) -> US
    - Common aliases (UK, USA, England) -> proper codes

    Returns empty string if region cannot be normalized.
    """
    if not region:
        return ""

    # Check cache first
    cache_key = region.lower().strip()
    if cache_key in _region_cache:
        return _region_cache[cache_key]

    result = _normalize_region_uncached(region)
    _region_cache[cache_key] = result
    return result


def _normalize_region_uncached(region: str) -> str:
    """Uncached region normalization logic."""
    region_clean = region.strip()

    # Empty after stripping = empty result
    if not region_clean:
        return ""

    region_lower = region_clean.lower()
    region_upper = region_clean.upper()

    # Check common aliases first
    if region_lower in REGION_ALIASES:
        return REGION_ALIASES[region_lower]

    # For 2-letter codes, check country first, then US state
    # This means ambiguous codes like "CA" (Canada vs California) prefer country
    # But unambiguous codes like "NY" (not a country) will match as US state
    if len(region_clean) == 2:
        # Try as country alpha-2 first
        country = pycountry.countries.get(alpha_2=region_upper)
        if country:
            return country.alpha_2

        # If not a country, try as US state code
        subdivision = pycountry.subdivisions.get(code=f"US-{region_upper}")
        if subdivision:
            return "US"

    # Try alpha-3 lookup
    if len(region_clean) == 3:
        country = pycountry.countries.get(alpha_3=region_upper)
        if country:
            return country.alpha_2

    # Try as US state name (e.g., "California", "New York")
    try:
        subdivisions = list(pycountry.subdivisions.search_fuzzy(region_clean))
        if subdivisions:
            # Check if it's a US state
            if subdivisions[0].code.startswith("US-"):
                return "US"
            # Return the parent country code
            return subdivisions[0].country_code
    except LookupError:
        pass

    # Try country fuzzy search
    try:
        countries = pycountry.countries.search_fuzzy(region_clean)
        if countries:
            return countries[0].alpha_2
    except LookupError:
        pass

    # Return empty if we can't normalize
    return ""


def _regions_match(region1: str, region2: str) -> bool:
    """
    Check if two regions match after normalization.

    Empty regions match anything (lenient matching for incomplete data).
    """
    norm1 = _normalize_region(region1)
    norm2 = _normalize_region(region2)

    # Empty regions match anything
    if not norm1 or not norm2:
        return True

    return norm1 == norm2


def _normalize_for_canon(name: str) -> str:
    """Normalize name for canonical matching (simpler than search normalization)."""
    # Lowercase
    result = name.lower()
    # Remove trailing dots
    result = result.rstrip(".")
    # Remove extra whitespace
    result = " ".join(result.split())
    return result


def _expand_suffix(name: str) -> str:
    """Expand known suffix abbreviations."""
    result = name.lower().rstrip(".")
    for abbrev, full in SUFFIX_EXPANSIONS.items():
        if result.endswith(abbrev):
            result = result[:-len(abbrev)] + full
            break  # Only expand one suffix
    return result


def _names_match_for_canon(name1: str, name2: str) -> bool:
    """Check if two names match for canonicalization."""
    n1 = _normalize_for_canon(name1)
    n2 = _normalize_for_canon(name2)

    # Exact match after normalization
    if n1 == n2:
        return True

    # Try with suffix expansion
    if _expand_suffix(n1) == _expand_suffix(n2):
        return True

    return False

# Pre-compile the suffix pattern for performance
_SUFFIX_PATTERN = re.compile(
    r'\s+(' + '|'.join(re.escape(suffix) for suffix in COMPANY_SUFFIXES) + r')\.?$',
    re.IGNORECASE
)


def _clean_org_name(name: str | None) -> str:
    """
    Remove special characters and formatting from organization name.

    Removes brackets, parentheses, quotes, and other formatting artifacts.
    """
    if not name:
        return ""
    # Remove special characters, keeping only alphanumeric and spaces
    cleaned = re.sub(r'[•;:\'"\[\](){}<>`~!@#$%^&*\-_=+\\|/?!`~]+', ' ', name)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    # Recurse if changes were made (handles nested special chars)
    return _clean_org_name(cleaned) if cleaned != name else cleaned


def _remove_suffix(name: str) -> str:
    """
    Remove corporate legal suffixes from company name.

    Iteratively removes suffixes until no more are found.
    Also removes possessive 's and trailing punctuation.
    """
    cleaned = name.strip()
    cleaned = re.sub(r'\s+', ' ', cleaned)
    # Remove possessive 's (e.g., "Amazon's" -> "Amazon")
    cleaned = re.sub(r"'s\b", "", cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()

    while True:
        new_name = _SUFFIX_PATTERN.sub('', cleaned)
        # Remove trailing punctuation
        new_name = re.sub(r'[ .,;&\n\t/)]$', '', new_name)

        if new_name == cleaned:
            break
        cleaned = new_name.strip()

    return cleaned.strip()


def _normalize_name(name: str) -> str:
    """
    Normalize company name for text matching.

    1. Remove possessive 's (before cleaning removes apostrophe)
    2. Clean special characters
    3. Remove legal suffixes
    4. Lowercase
    5. If result is empty, use cleaned lowercase original

    Always returns a non-empty string for valid input.
    """
    if not name:
        return ""
    # Remove possessive 's first (before cleaning removes the apostrophe)
    normalized = re.sub(r"'s\b", "", name)
    # Clean special characters
    cleaned = _clean_org_name(normalized)
    # Remove legal suffixes
    normalized = _remove_suffix(cleaned)
    # Lowercase for matching
    normalized = normalized.lower()
    # If normalized is empty (e.g., name was just "Ltd"), use the cleaned name
    if not normalized:
        normalized = cleaned.lower() if cleaned else name.lower()
    return normalized


def _extract_search_terms(query: str) -> list[str]:
    """
    Extract search terms from a query for SQL LIKE matching.

    Returns list of terms to search for, ordered by length (longest first).
    """
    # Split into words
    words = query.split()

    # Filter out very short words (< 3 chars) unless it's the only word
    if len(words) > 1:
        words = [w for w in words if len(w) >= 3]

    # Sort by length descending (longer words are more specific)
    words.sort(key=len, reverse=True)

    return words[:3]  # Limit to top 3 terms


# Person name normalization patterns
_PERSON_PREFIXES = {
    "dr.", "dr", "prof.", "prof", "professor",
    "mr.", "mr", "mrs.", "mrs", "ms.", "ms", "miss",
    "sir", "dame", "lord", "lady",
    "rev.", "rev", "reverend",
    "hon.", "hon", "honorable",
    "gen.", "gen", "general",
    "col.", "col", "colonel",
    "capt.", "capt", "captain",
    "lt.", "lt", "lieutenant",
    "sgt.", "sgt", "sergeant",
}

_PERSON_SUFFIXES = {
    "jr.", "jr", "junior",
    "sr.", "sr", "senior",
    "ii", "iii", "iv", "v",
    "2nd", "3rd", "4th", "5th",
    "phd", "ph.d.", "ph.d",
    "md", "m.d.", "m.d",
    "esq", "esq.",
    "mba", "m.b.a.",
    "cpa", "c.p.a.",
    "jd", "j.d.",
}


def _normalize_person_name(name: str) -> str:
    """
    Normalize person name for text matching.

    1. Remove honorific prefixes (Dr., Prof., Mr., etc.)
    2. Remove generational suffixes (Jr., Sr., III, PhD, etc.)
    3. Keep name particles (von, van, de, al-, etc.)
    4. Lowercase and strip

    Always returns a non-empty string for valid input.
    """
    if not name:
        return ""

    # Lowercase for matching
    normalized = name.lower().strip()

    # Split into words
    words = normalized.split()
    if not words:
        return ""

    # Remove prefix if first word is a title
    while words and words[0].rstrip(".") in _PERSON_PREFIXES:
        words.pop(0)
        if not words:
            return name.lower().strip()  # Fallback if name was just a title

    # Remove suffix if last word is a suffix
    while words and words[-1].rstrip(".") in _PERSON_SUFFIXES:
        words.pop()
        if not words:
            return name.lower().strip()  # Fallback if name was just suffixes

    # Rejoin remaining words
    normalized = " ".join(words)

    # Clean up extra spaces
    normalized = re.sub(r'\s+', ' ', normalized).strip()

    return normalized if normalized else name.lower().strip()


def get_database(db_path: Optional[str | Path] = None, embedding_dim: int = 768) -> "OrganizationDatabase":
    """
    Get a singleton OrganizationDatabase instance for the given path.

    Args:
        db_path: Path to database file
        embedding_dim: Dimension of embeddings

    Returns:
        Shared OrganizationDatabase instance
    """
    path_key = str(db_path or DEFAULT_DB_PATH)
    if path_key not in _database_instances:
        logger.debug(f"Creating new OrganizationDatabase instance for {path_key}")
        _database_instances[path_key] = OrganizationDatabase(db_path=db_path, embedding_dim=embedding_dim)
    return _database_instances[path_key]


class OrganizationDatabase:
    """
    SQLite database with sqlite-vec for organization vector search.

    Uses hybrid text + vector search:
    1. Text filtering with Levenshtein distance to reduce candidates
    2. sqlite-vec for semantic similarity ranking
    """

    def __init__(
        self,
        db_path: Optional[str | Path] = None,
        embedding_dim: int = 768,  # Default for embeddinggemma-300m
    ):
        """
        Initialize the organization database.

        Args:
            db_path: Path to database file (creates if not exists)
            embedding_dim: Dimension of embeddings to store
        """
        self._db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self._embedding_dim = embedding_dim
        self._conn: Optional[sqlite3.Connection] = None

    def _ensure_dir(self) -> None:
        """Ensure database directory exists."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

    def _connect(self) -> sqlite3.Connection:
        """Get or create database connection using shared connection pool."""
        if self._conn is not None:
            return self._conn

        self._conn = _get_shared_connection(self._db_path, self._embedding_dim)

        # Create tables (idempotent)
        self._create_tables()

        return self._conn

    def _create_tables(self) -> None:
        """Create database tables including sqlite-vec virtual table."""
        conn = self._conn
        assert conn is not None

        # Main organization records table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS organizations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                name_normalized TEXT NOT NULL,
                source TEXT NOT NULL,
                source_id TEXT NOT NULL,
                region TEXT NOT NULL DEFAULT '',
                entity_type TEXT NOT NULL DEFAULT 'unknown',
                from_date TEXT NOT NULL DEFAULT '',
                to_date TEXT NOT NULL DEFAULT '',
                record TEXT NOT NULL,
                UNIQUE(source, source_id)
            )
        """)

        # Add region column if it doesn't exist (migration for existing DBs)
        try:
            conn.execute("ALTER TABLE organizations ADD COLUMN region TEXT NOT NULL DEFAULT ''")
            logger.info("Added region column to organizations table")
        except sqlite3.OperationalError:
            pass  # Column already exists

        # Add entity_type column if it doesn't exist (migration for existing DBs)
        try:
            conn.execute("ALTER TABLE organizations ADD COLUMN entity_type TEXT NOT NULL DEFAULT 'unknown'")
            logger.info("Added entity_type column to organizations table")
        except sqlite3.OperationalError:
            pass  # Column already exists

        # Add from_date column if it doesn't exist (migration for existing DBs)
        try:
            conn.execute("ALTER TABLE organizations ADD COLUMN from_date TEXT NOT NULL DEFAULT ''")
            logger.info("Added from_date column to organizations table")
        except sqlite3.OperationalError:
            pass  # Column already exists

        # Add to_date column if it doesn't exist (migration for existing DBs)
        try:
            conn.execute("ALTER TABLE organizations ADD COLUMN to_date TEXT NOT NULL DEFAULT ''")
            logger.info("Added to_date column to organizations table")
        except sqlite3.OperationalError:
            pass  # Column already exists

        # Add canon_id column if it doesn't exist (migration for canonicalization)
        try:
            conn.execute("ALTER TABLE organizations ADD COLUMN canon_id INTEGER DEFAULT NULL")
            logger.info("Added canon_id column to organizations table")
        except sqlite3.OperationalError:
            pass  # Column already exists

        # Add canon_size column if it doesn't exist (migration for canonicalization)
        try:
            conn.execute("ALTER TABLE organizations ADD COLUMN canon_size INTEGER DEFAULT 1")
            logger.info("Added canon_size column to organizations table")
        except sqlite3.OperationalError:
            pass  # Column already exists

        # Create indexes on main table
        conn.execute("CREATE INDEX IF NOT EXISTS idx_orgs_name ON organizations(name)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_orgs_name_normalized ON organizations(name_normalized)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_orgs_source ON organizations(source)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_orgs_source_id ON organizations(source, source_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_orgs_region ON organizations(region)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_orgs_entity_type ON organizations(entity_type)")
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_orgs_name_region_source ON organizations(name, region, source)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_orgs_canon_id ON organizations(canon_id)")

        # Create sqlite-vec virtual table for embeddings
        # vec0 is the recommended virtual table type
        conn.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS organization_embeddings USING vec0(
                org_id INTEGER PRIMARY KEY,
                embedding float[{self._embedding_dim}]
            )
        """)

        conn.commit()

    def close(self) -> None:
        """Clear connection reference (shared connection remains open)."""
        self._conn = None

    def insert(self, record: CompanyRecord, embedding: np.ndarray) -> int:
        """
        Insert an organization record with its embedding.

        Args:
            record: Organization record to insert
            embedding: Embedding vector for the organization name

        Returns:
            Row ID of inserted record
        """
        conn = self._connect()

        # Serialize record
        record_json = json.dumps(record.record)
        name_normalized = _normalize_name(record.name)

        cursor = conn.execute("""
            INSERT OR REPLACE INTO organizations
            (name, name_normalized, source, source_id, region, entity_type, from_date, to_date, record)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            record.name,
            name_normalized,
            record.source,
            record.source_id,
            record.region,
            record.entity_type.value,
            record.from_date or "",
            record.to_date or "",
            record_json,
        ))

        row_id = cursor.lastrowid
        assert row_id is not None

        # Insert embedding into vec table
        # sqlite-vec virtual tables don't support INSERT OR REPLACE, so delete first
        embedding_blob = embedding.astype(np.float32).tobytes()
        conn.execute("DELETE FROM organization_embeddings WHERE org_id = ?", (row_id,))
        conn.execute("""
            INSERT INTO organization_embeddings (org_id, embedding)
            VALUES (?, ?)
        """, (row_id, embedding_blob))

        conn.commit()
        return row_id

    def insert_batch(
        self,
        records: list[CompanyRecord],
        embeddings: np.ndarray,
        batch_size: int = 1000,
    ) -> int:
        """
        Insert multiple organization records with embeddings.

        Args:
            records: List of organization records
            embeddings: Matrix of embeddings (N x dim)
            batch_size: Commit batch size

        Returns:
            Number of records inserted
        """
        conn = self._connect()
        count = 0

        for record, embedding in zip(records, embeddings):
            record_json = json.dumps(record.record)
            name_normalized = _normalize_name(record.name)

            cursor = conn.execute("""
                INSERT OR REPLACE INTO organizations
                (name, name_normalized, source, source_id, region, entity_type, from_date, to_date, record)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.name,
                name_normalized,
                record.source,
                record.source_id,
                record.region,
                record.entity_type.value,
                record.from_date or "",
                record.to_date or "",
                record_json,
            ))

            row_id = cursor.lastrowid
            assert row_id is not None

            # Insert embedding (delete first since sqlite-vec doesn't support REPLACE)
            embedding_blob = embedding.astype(np.float32).tobytes()
            conn.execute("DELETE FROM organization_embeddings WHERE org_id = ?", (row_id,))
            conn.execute("""
                INSERT INTO organization_embeddings (org_id, embedding)
                VALUES (?, ?)
            """, (row_id, embedding_blob))

            count += 1

            if count % batch_size == 0:
                conn.commit()
                logger.info(f"Inserted {count} records...")

        conn.commit()
        return count

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 20,
        source_filter: Optional[str] = None,
        query_text: Optional[str] = None,
        max_text_candidates: int = 5000,
        rerank_min_candidates: int = 500,
    ) -> list[tuple[CompanyRecord, float]]:
        """
        Search for similar organizations using hybrid text + vector search.

        Three-stage approach:
        1. If query_text provided, use SQL LIKE to find candidates containing search terms
        2. Use sqlite-vec for vector similarity ranking on filtered candidates
        3. Apply prominence-based re-ranking to boost major companies (SEC filers, tickers)

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            source_filter: Optional filter by source (gleif, sec_edgar, etc.)
            query_text: Optional query text for text-based pre-filtering
            max_text_candidates: Max candidates to keep after text filtering
            rerank_min_candidates: Minimum candidates to fetch for re-ranking (default 500)

        Returns:
            List of (CompanyRecord, adjusted_score) tuples sorted by prominence-adjusted score
        """
        start = time.time()
        self._connect()

        # Normalize query embedding
        query_norm = np.linalg.norm(query_embedding)
        if query_norm == 0:
            return []
        query_normalized = query_embedding / query_norm
        query_blob = query_normalized.astype(np.float32).tobytes()

        # Stage 1: Text-based pre-filtering (if query_text provided)
        candidate_ids: Optional[set[int]] = None
        query_normalized_text = ""
        if query_text:
            query_normalized_text = _normalize_name(query_text)
            if query_normalized_text:
                candidate_ids = self._text_filter_candidates(
                    query_normalized_text,
                    max_candidates=max_text_candidates,
                    source_filter=source_filter,
                )
                logger.info(f"Text filter: {len(candidate_ids)} candidates for '{query_text}'")

        # Stage 2: Vector search - fetch more candidates for re-ranking
        if candidate_ids is not None and len(candidate_ids) == 0:
            # No text matches, return empty
            return []

        # Fetch enough candidates for prominence re-ranking to be effective
        # Use at least rerank_min_candidates, or all text-filtered candidates if fewer
        if candidate_ids is not None:
            fetch_k = min(len(candidate_ids), max(rerank_min_candidates, top_k * 5))
        else:
            fetch_k = max(rerank_min_candidates, top_k * 5)

        if candidate_ids is not None:
            # Search within text-filtered candidates
            results = self._vector_search_filtered(
                query_blob, candidate_ids, fetch_k, source_filter
            )
        else:
            # Full vector search
            results = self._vector_search_full(query_blob, fetch_k, source_filter)

        # Stage 3: Prominence-based re-ranking
        if results and query_normalized_text:
            results = self._apply_prominence_reranking(results, query_normalized_text, top_k)
        else:
            # No re-ranking, just trim to top_k
            results = results[:top_k]

        elapsed = time.time() - start
        logger.debug(f"Hybrid search took {elapsed:.3f}s (results={len(results)})")
        return results

    def _calculate_prominence_boost(
        self,
        record: CompanyRecord,
        query_normalized: str,
        canon_sources: Optional[set[str]] = None,
    ) -> float:
        """
        Calculate prominence boost for re-ranking search results.

        Boosts scores based on signals that indicate a major/prominent company:
        - Has ticker symbol (publicly traded)
        - GLEIF source (has LEI)
        - SEC source (vetted US filers)
        - Wikidata source (Wikipedia-notable)
        - Exact normalized name match

        When canon_sources is provided (from a canonical group), boosts are
        applied for ALL sources in the canon group, not just this record's source.

        Args:
            record: The company record to evaluate
            query_normalized: Normalized query text for exact match check
            canon_sources: Optional set of sources in this record's canonical group

        Returns:
            Boost value to add to embedding similarity (0.0 to ~0.21)
        """
        boost = 0.0

        # Get all sources to consider (canon group or just this record)
        sources_to_check = canon_sources or {record.source}

        # Has ticker symbol = publicly traded major company
        # Check if ANY record in canon group has ticker
        if record.record.get("ticker") or (canon_sources and "sec_edgar" in canon_sources):
            boost += 0.08

        # Source-based boosts - accumulate for all sources in canon group
        if "gleif" in sources_to_check:
            boost += 0.05  # Has LEI = verified legal entity
        if "sec_edgar" in sources_to_check:
            boost += 0.03  # SEC filer
        if "wikipedia" in sources_to_check:
            boost += 0.02  # Wikipedia notable

        # Exact normalized name match bonus
        record_normalized = _normalize_name(record.name)
        if query_normalized == record_normalized:
            boost += 0.05

        return boost

    def _apply_prominence_reranking(
        self,
        results: list[tuple[CompanyRecord, float]],
        query_normalized: str,
        top_k: int,
        similarity_weight: float = 0.3,
    ) -> list[tuple[CompanyRecord, float]]:
        """
        Apply prominence-based re-ranking to search results with canon group awareness.

        When records have been canonicalized, boosts are applied based on ALL sources
        in the canonical group, not just the matched record's source.

        Args:
            results: List of (record, similarity) from vector search
            query_normalized: Normalized query text
            top_k: Number of results to return after re-ranking
            similarity_weight: Weight for similarity score (0-1), lower = prominence matters more

        Returns:
            Re-ranked list of (record, adjusted_score) tuples
        """
        conn = self._conn
        assert conn is not None

        # Build canon_id -> sources mapping for all results that have canon_id
        canon_sources_map: dict[int, set[str]] = {}
        canon_ids = [
            r.record.get("canon_id")
            for r, _ in results
            if r.record.get("canon_id") is not None
        ]

        if canon_ids:
            # Fetch all sources for each canon_id in one query
            unique_canon_ids = list(set(canon_ids))
            placeholders = ",".join("?" * len(unique_canon_ids))
            rows = conn.execute(f"""
                SELECT canon_id, source
                FROM organizations
                WHERE canon_id IN ({placeholders})
            """, unique_canon_ids).fetchall()

            for row in rows:
                canon_id = row["canon_id"]
                canon_sources_map.setdefault(canon_id, set()).add(row["source"])

        # Calculate boosted scores with canon group awareness
        # Formula: adjusted = (similarity * weight) + boost
        # With weight=0.3, a sim=0.65 SEC+ticker (boost=0.11) beats sim=0.75 no-boost
        boosted_results: list[tuple[CompanyRecord, float, float, float]] = []
        for record, similarity in results:
            canon_id = record.record.get("canon_id")
            # Get all sources in this record's canon group (if any)
            canon_sources = canon_sources_map.get(canon_id) if canon_id else None

            boost = self._calculate_prominence_boost(record, query_normalized, canon_sources)
            adjusted_score = (similarity * similarity_weight) + boost
            boosted_results.append((record, similarity, boost, adjusted_score))

        # Sort by adjusted score (descending)
        boosted_results.sort(key=lambda x: x[3], reverse=True)

        # Log re-ranking details for top results
        logger.debug(f"Prominence re-ranking for '{query_normalized}':")
        for record, sim, boost, adj in boosted_results[:10]:
            ticker = record.record.get("ticker", "")
            ticker_str = f" ticker={ticker}" if ticker else ""
            canon_id = record.record.get("canon_id")
            canon_str = f" canon={canon_id}" if canon_id else ""
            logger.debug(
                f"  {record.name}: sim={sim:.3f} + boost={boost:.3f} = {adj:.3f} "
                f"[{record.source}{ticker_str}{canon_str}]"
            )

        # Return top_k with adjusted scores
        return [(r, adj) for r, _, _, adj in boosted_results[:top_k]]

    def _text_filter_candidates(
        self,
        query_normalized: str,
        max_candidates: int,
        source_filter: Optional[str] = None,
    ) -> set[int]:
        """
        Filter candidates using SQL LIKE for fast text matching.

        This is a generous pre-filter to reduce the embedding search space.
        Returns set of organization IDs that contain any search term.
        Uses `name_normalized` column for consistent matching.
        """
        conn = self._conn
        assert conn is not None

        # Extract search terms from the normalized query
        search_terms = _extract_search_terms(query_normalized)
        if not search_terms:
            return set()

        logger.debug(f"Text filter search terms: {search_terms}")

        # Build OR clause for LIKE matching on any term
        # Use name_normalized for consistent matching (already lowercased, suffixes removed)
        like_clauses = []
        params: list = []
        for term in search_terms:
            like_clauses.append("name_normalized LIKE ?")
            params.append(f"%{term}%")

        where_clause = " OR ".join(like_clauses)

        # Add source filter if specified
        if source_filter:
            query = f"""
                SELECT id FROM organizations
                WHERE ({where_clause}) AND source = ?
                LIMIT ?
            """
            params.append(source_filter)
        else:
            query = f"""
                SELECT id FROM organizations
                WHERE {where_clause}
                LIMIT ?
            """

        params.append(max_candidates)

        cursor = conn.execute(query, params)
        return set(row["id"] for row in cursor)

    def _vector_search_filtered(
        self,
        query_blob: bytes,
        candidate_ids: set[int],
        top_k: int,
        source_filter: Optional[str],
    ) -> list[tuple[CompanyRecord, float]]:
        """Vector search within a filtered set of candidates."""
        conn = self._conn
        assert conn is not None

        if not candidate_ids:
            return []

        # Build IN clause for candidate IDs
        placeholders = ",".join("?" * len(candidate_ids))

        # Query sqlite-vec with KNN search, filtered by candidate IDs
        # Using distance function - lower is more similar for L2
        # We'll use cosine distance
        query = f"""
            SELECT
                e.org_id,
                vec_distance_cosine(e.embedding, ?) as distance
            FROM organization_embeddings e
            WHERE e.org_id IN ({placeholders})
            ORDER BY distance
            LIMIT ?
        """

        cursor = conn.execute(query, [query_blob] + list(candidate_ids) + [top_k])

        results = []
        for row in cursor:
            org_id = row["org_id"]
            distance = row["distance"]
            # Convert cosine distance to similarity (1 - distance)
            similarity = 1.0 - distance

            # Fetch full record
            record = self._get_record_by_id(org_id)
            if record:
                # Apply source filter if specified
                if source_filter and record.source != source_filter:
                    continue
                results.append((record, similarity))

        return results

    def _vector_search_full(
        self,
        query_blob: bytes,
        top_k: int,
        source_filter: Optional[str],
    ) -> list[tuple[CompanyRecord, float]]:
        """Full vector search without text pre-filtering."""
        conn = self._conn
        assert conn is not None

        # KNN search with sqlite-vec
        if source_filter:
            # Need to join with organizations table for source filter
            query = """
                SELECT
                    e.org_id,
                    vec_distance_cosine(e.embedding, ?) as distance
                FROM organization_embeddings e
                JOIN organizations c ON e.org_id = c.id
                WHERE c.source = ?
                ORDER BY distance
                LIMIT ?
            """
            cursor = conn.execute(query, (query_blob, source_filter, top_k))
        else:
            query = """
                SELECT
                    org_id,
                    vec_distance_cosine(embedding, ?) as distance
                FROM organization_embeddings
                ORDER BY distance
                LIMIT ?
            """
            cursor = conn.execute(query, (query_blob, top_k))

        results = []
        for row in cursor:
            org_id = row["org_id"]
            distance = row["distance"]
            similarity = 1.0 - distance

            record = self._get_record_by_id(org_id)
            if record:
                results.append((record, similarity))

        return results

    def _get_record_by_id(self, org_id: int) -> Optional[CompanyRecord]:
        """Get an organization record by ID, including db_id and canon_id in record dict."""
        conn = self._conn
        assert conn is not None

        cursor = conn.execute("""
            SELECT id, name, source, source_id, region, entity_type, record, canon_id
            FROM organizations WHERE id = ?
        """, (org_id,))

        row = cursor.fetchone()
        if row:
            record_data = json.loads(row["record"])
            # Add db_id and canon_id to record dict for canon-aware search
            record_data["db_id"] = row["id"]
            record_data["canon_id"] = row["canon_id"]
            return CompanyRecord(
                name=row["name"],
                source=row["source"],
                source_id=row["source_id"],
                region=row["region"] or "",
                entity_type=EntityType(row["entity_type"]) if row["entity_type"] else EntityType.UNKNOWN,
                record=record_data,
            )
        return None

    def get_by_source_id(self, source: str, source_id: str) -> Optional[CompanyRecord]:
        """Get an organization record by source and source_id."""
        conn = self._connect()

        cursor = conn.execute("""
            SELECT name, source, source_id, region, entity_type, record
            FROM organizations
            WHERE source = ? AND source_id = ?
        """, (source, source_id))

        row = cursor.fetchone()
        if row:
            return CompanyRecord(
                name=row["name"],
                source=row["source"],
                source_id=row["source_id"],
                region=row["region"] or "",
                entity_type=EntityType(row["entity_type"]) if row["entity_type"] else EntityType.UNKNOWN,
                record=json.loads(row["record"]),
            )
        return None

    def get_id_by_source_id(self, source: str, source_id: str) -> Optional[int]:
        """Get the internal database ID for an organization by source and source_id."""
        conn = self._connect()

        cursor = conn.execute("""
            SELECT id FROM organizations
            WHERE source = ? AND source_id = ?
        """, (source, source_id))

        row = cursor.fetchone()
        if row:
            return row["id"]
        return None

    def get_stats(self) -> DatabaseStats:
        """Get database statistics."""
        conn = self._connect()

        # Total count
        cursor = conn.execute("SELECT COUNT(*) FROM organizations")
        total = cursor.fetchone()[0]

        # Count by source
        cursor = conn.execute("SELECT source, COUNT(*) as cnt FROM organizations GROUP BY source")
        by_source = {row["source"]: row["cnt"] for row in cursor}

        # Database file size
        db_size = self._db_path.stat().st_size if self._db_path.exists() else 0

        return DatabaseStats(
            total_records=total,
            by_source=by_source,
            embedding_dimension=self._embedding_dim,
            database_size_bytes=db_size,
        )

    def get_all_source_ids(self, source: Optional[str] = None) -> set[str]:
        """
        Get all source_ids from the organizations table.

        Useful for resume operations to skip already-imported records.

        Args:
            source: Optional source filter (e.g., "wikipedia" for Wikidata orgs)

        Returns:
            Set of source_id strings (e.g., Q codes for Wikidata)
        """
        conn = self._connect()

        if source:
            cursor = conn.execute(
                "SELECT DISTINCT source_id FROM organizations WHERE source = ?",
                (source,)
            )
        else:
            cursor = conn.execute("SELECT DISTINCT source_id FROM organizations")

        return {row[0] for row in cursor}

    def iter_records(self, source: Optional[str] = None) -> Iterator[CompanyRecord]:
        """Iterate over all records, optionally filtered by source."""
        conn = self._connect()

        if source:
            cursor = conn.execute("""
                SELECT name, source, source_id, region, entity_type, record
                FROM organizations
                WHERE source = ?
            """, (source,))
        else:
            cursor = conn.execute("""
                SELECT name, source, source_id, region, entity_type, record
                FROM organizations
            """)

        for row in cursor:
            yield CompanyRecord(
                name=row["name"],
                source=row["source"],
                source_id=row["source_id"],
                region=row["region"] or "",
                entity_type=EntityType(row["entity_type"]) if row["entity_type"] else EntityType.UNKNOWN,
                record=json.loads(row["record"]),
            )

    def canonicalize(self, batch_size: int = 10000) -> dict[str, int]:
        """
        Canonicalize all organizations by linking equivalent records.

        Records are considered equivalent if they match by:
        1. Same LEI (GLEIF source_id or Wikidata P1278) - globally unique, no region check
        2. Same ticker symbol - globally unique, no region check
        3. Same CIK - globally unique, no region check
        4. Same normalized name AND same normalized region
        5. Name match with suffix expansion AND same region

        Region normalization uses pycountry to handle:
        - Country codes/names (GB, United Kingdom, Great Britain -> GB)
        - US state codes/names (CA, California -> US)
        - Common aliases (UK -> GB, USA -> US)

        For each group of equivalent records, the highest-priority source
        (gleif > sec_edgar > companies_house > wikipedia) becomes canonical.

        Args:
            batch_size: Commit batch size for updates

        Returns:
            Dict with stats: total_records, groups_found, records_updated
        """
        conn = self._connect()
        logger.info("Starting canonicalization...")

        # Phase 1: Load all organization data and build indexes
        logger.info("Phase 1: Building indexes...")

        lei_index: dict[str, list[int]] = {}
        ticker_index: dict[str, list[int]] = {}
        cik_index: dict[str, list[int]] = {}
        # Name indexes now keyed by (normalized_name, normalized_region)
        # Region-less matching only applies for identifier-based matching
        name_region_index: dict[tuple[str, str], list[int]] = {}
        expanded_name_region_index: dict[tuple[str, str], list[int]] = {}

        sources: dict[int, str] = {}  # org_id -> source
        all_org_ids: list[int] = []

        cursor = conn.execute("""
            SELECT id, source, source_id, name, region, record
            FROM organizations
        """)

        count = 0
        for row in cursor:
            org_id = row["id"]
            source = row["source"]
            name = row["name"]
            region = row["region"] or ""
            record = json.loads(row["record"])

            all_org_ids.append(org_id)
            sources[org_id] = source

            # Index by LEI (GLEIF source_id or Wikidata's P1278)
            # LEI is globally unique - no region check needed
            if source == "gleif":
                lei = row["source_id"]
            else:
                lei = record.get("lei")
            if lei:
                lei_index.setdefault(lei.upper(), []).append(org_id)

            # Index by ticker - globally unique, no region check
            ticker = record.get("ticker")
            if ticker:
                ticker_index.setdefault(ticker.upper(), []).append(org_id)

            # Index by CIK - globally unique, no region check
            if source == "sec_edgar":
                cik = row["source_id"]
            else:
                cik = record.get("cik")
            if cik:
                cik_index.setdefault(str(cik), []).append(org_id)

            # Index by (normalized_name, normalized_region)
            # Same name in different regions = different legal entities
            norm_name = _normalize_for_canon(name)
            norm_region = _normalize_region(region)
            if norm_name:
                key = (norm_name, norm_region)
                name_region_index.setdefault(key, []).append(org_id)

            # Index by (expanded_name, normalized_region)
            expanded_name = _expand_suffix(name)
            if expanded_name and expanded_name != norm_name:
                key = (expanded_name, norm_region)
                expanded_name_region_index.setdefault(key, []).append(org_id)

            count += 1
            if count % 100000 == 0:
                logger.info(f"  Indexed {count} organizations...")

        logger.info(f"  Indexed {count} organizations total")
        logger.info(f"  LEI index: {len(lei_index)} unique LEIs")
        logger.info(f"  Ticker index: {len(ticker_index)} unique tickers")
        logger.info(f"  CIK index: {len(cik_index)} unique CIKs")
        logger.info(f"  Name+region index: {len(name_region_index)} unique (name, region) pairs")
        logger.info(f"  Expanded name+region index: {len(expanded_name_region_index)} unique pairs")

        # Phase 2: Build equivalence groups using Union-Find
        logger.info("Phase 2: Building equivalence groups...")

        uf = UnionFind(all_org_ids)

        # Merge by LEI (globally unique identifier)
        for _lei, ids in lei_index.items():
            for i in range(1, len(ids)):
                uf.union(ids[0], ids[i])

        # Merge by ticker (globally unique identifier)
        for _ticker, ids in ticker_index.items():
            for i in range(1, len(ids)):
                uf.union(ids[0], ids[i])

        # Merge by CIK (globally unique identifier)
        for _cik, ids in cik_index.items():
            for i in range(1, len(ids)):
                uf.union(ids[0], ids[i])

        # Merge by (normalized_name, normalized_region)
        for _name_region, ids in name_region_index.items():
            for i in range(1, len(ids)):
                uf.union(ids[0], ids[i])

        # Merge by (expanded_name, normalized_region)
        # This connects "Amazon Ltd" with "Amazon Limited" in same region
        for key, expanded_ids in expanded_name_region_index.items():
            # Find org_ids with the expanded form as their normalized name in same region
            if key in name_region_index:
                # Link first expanded_id to first name_id
                uf.union(expanded_ids[0], name_region_index[key][0])

        groups = uf.groups()
        logger.info(f"  Found {len(groups)} equivalence groups")

        # Count groups with multiple records
        multi_record_groups = sum(1 for ids in groups.values() if len(ids) > 1)
        logger.info(f"  Groups with multiple records: {multi_record_groups}")

        # Phase 3: Select canonical record for each group and update database
        logger.info("Phase 3: Updating database...")

        updated_count = 0
        batch_updates: list[tuple[int, int, int]] = []  # (org_id, canon_id, canon_size)

        for _root, group_ids in groups.items():
            if len(group_ids) == 1:
                # Single record - canonical to itself
                batch_updates.append((group_ids[0], group_ids[0], 1))
            else:
                # Multiple records - find highest priority source
                best_id = min(
                    group_ids,
                    key=lambda oid: (SOURCE_PRIORITY.get(sources[oid], 99), oid)
                )
                group_size = len(group_ids)

                # All records in group point to the best one
                for oid in group_ids:
                    # canon_size is only set on the canonical record
                    size = group_size if oid == best_id else 1
                    batch_updates.append((oid, best_id, size))

            # Commit batch
            if len(batch_updates) >= batch_size:
                self._apply_canon_updates(batch_updates)
                updated_count += len(batch_updates)
                logger.info(f"  Updated {updated_count} records...")
                batch_updates = []

        # Final batch
        if batch_updates:
            self._apply_canon_updates(batch_updates)
            updated_count += len(batch_updates)

        conn.commit()
        logger.info(f"Canonicalization complete: {updated_count} records updated, {multi_record_groups} multi-record groups")

        return {
            "total_records": count,
            "groups_found": len(groups),
            "multi_record_groups": multi_record_groups,
            "records_updated": updated_count,
        }

    def _apply_canon_updates(self, updates: list[tuple[int, int, int]]) -> None:
        """Apply batch of canon updates: (org_id, canon_id, canon_size)."""
        conn = self._conn
        assert conn is not None

        for org_id, canon_id, canon_size in updates:
            conn.execute(
                "UPDATE organizations SET canon_id = ?, canon_size = ? WHERE id = ?",
                (canon_id, canon_size, org_id)
            )

        conn.commit()

    def get_canon_stats(self) -> dict[str, int]:
        """Get statistics about canonicalization status."""
        conn = self._connect()

        # Total records
        cursor = conn.execute("SELECT COUNT(*) FROM organizations")
        total = cursor.fetchone()[0]

        # Records with canon_id set
        cursor = conn.execute("SELECT COUNT(*) FROM organizations WHERE canon_id IS NOT NULL")
        canonicalized = cursor.fetchone()[0]

        # Number of canonical groups (unique canon_ids)
        cursor = conn.execute("SELECT COUNT(DISTINCT canon_id) FROM organizations WHERE canon_id IS NOT NULL")
        groups = cursor.fetchone()[0]

        # Multi-record groups (canon_size > 1)
        cursor = conn.execute("SELECT COUNT(*) FROM organizations WHERE canon_size > 1")
        multi_record_groups = cursor.fetchone()[0]

        # Records in multi-record groups
        cursor = conn.execute("""
            SELECT COUNT(*) FROM organizations o1
            WHERE EXISTS (SELECT 1 FROM organizations o2 WHERE o2.id = o1.canon_id AND o2.canon_size > 1)
        """)
        records_in_multi = cursor.fetchone()[0]

        return {
            "total_records": total,
            "canonicalized_records": canonicalized,
            "canonical_groups": groups,
            "multi_record_groups": multi_record_groups,
            "records_in_multi_groups": records_in_multi,
        }

    def migrate_name_normalized(self, batch_size: int = 50000) -> int:
        """
        Populate the name_normalized column for all records.

        This is a one-time migration for databases that don't have
        normalized names populated.

        Args:
            batch_size: Number of records to process per batch

        Returns:
            Number of records updated
        """
        conn = self._connect()

        # Check how many need migration (empty, null, or placeholder "-")
        cursor = conn.execute(
            "SELECT COUNT(*) FROM organizations WHERE name_normalized = '' OR name_normalized IS NULL OR name_normalized = '-'"
        )
        empty_count = cursor.fetchone()[0]

        if empty_count == 0:
            logger.info("All records already have name_normalized populated")
            return 0

        logger.info(f"Populating name_normalized for {empty_count} records...")

        updated = 0
        last_id = 0

        while True:
            # Get batch of records that need normalization, ordered by ID
            cursor = conn.execute("""
                SELECT id, name FROM organizations
                WHERE id > ? AND (name_normalized = '' OR name_normalized IS NULL OR name_normalized = '-')
                ORDER BY id
                LIMIT ?
            """, (last_id, batch_size))

            rows = cursor.fetchall()
            if not rows:
                break

            # Update each record
            for row in rows:
                # _normalize_name now always returns non-empty for valid input
                normalized = _normalize_name(row["name"])
                conn.execute(
                    "UPDATE organizations SET name_normalized = ? WHERE id = ?",
                    (normalized, row["id"])
                )
                last_id = row["id"]

            conn.commit()
            updated += len(rows)
            logger.info(f"  Updated {updated}/{empty_count} records...")

        logger.info(f"Migration complete: {updated} name_normalized values populated")
        return updated

    def migrate_to_sqlite_vec(self, batch_size: int = 10000) -> int:
        """
        Migrate embeddings from BLOB column to sqlite-vec virtual table.

        This is a one-time migration for databases created before sqlite-vec support.

        Args:
            batch_size: Number of records to process per batch

        Returns:
            Number of embeddings migrated
        """
        conn = self._connect()

        # Check if migration is needed
        cursor = conn.execute("SELECT COUNT(*) FROM organization_embeddings")
        vec_count = cursor.fetchone()[0]

        cursor = conn.execute("SELECT COUNT(*) FROM organizations WHERE embedding IS NOT NULL")
        blob_count = cursor.fetchone()[0]

        if vec_count >= blob_count:
            logger.info(f"Migration not needed: sqlite-vec has {vec_count} embeddings, BLOB has {blob_count}")
            return 0

        logger.info(f"Migrating {blob_count} embeddings from BLOB to sqlite-vec...")

        # Get IDs that need migration (in sqlite-vec but not in organizations)
        cursor = conn.execute("""
            SELECT c.id, c.embedding
            FROM organizations c
            LEFT JOIN organization_embeddings e ON c.id = e.org_id
            WHERE c.embedding IS NOT NULL AND e.org_id IS NULL
        """)

        migrated = 0
        batch = []

        for row in cursor:
            org_id = row["id"]
            embedding_blob = row["embedding"]

            if embedding_blob:
                batch.append((org_id, embedding_blob))

            if len(batch) >= batch_size:
                self._insert_vec_batch(batch)
                migrated += len(batch)
                logger.info(f"  Migrated {migrated}/{blob_count} embeddings...")
                batch = []

        # Insert remaining batch
        if batch:
            self._insert_vec_batch(batch)
            migrated += len(batch)

        logger.info(f"Migration complete: {migrated} embeddings migrated to sqlite-vec")
        return migrated

    def _insert_vec_batch(self, batch: list[tuple[int, bytes]]) -> None:
        """Insert a batch of embeddings into sqlite-vec table."""
        conn = self._conn
        assert conn is not None

        for org_id, embedding_blob in batch:
            conn.execute("DELETE FROM organization_embeddings WHERE org_id = ?", (org_id,))
            conn.execute("""
                INSERT INTO organization_embeddings (org_id, embedding)
                VALUES (?, ?)
            """, (org_id, embedding_blob))

        conn.commit()

    def delete_source(self, source: str) -> int:
        """Delete all records from a specific source."""
        conn = self._connect()

        # First get IDs to delete from vec table
        cursor = conn.execute("SELECT id FROM organizations WHERE source = ?", (source,))
        ids_to_delete = [row["id"] for row in cursor]

        # Delete from vec table
        if ids_to_delete:
            placeholders = ",".join("?" * len(ids_to_delete))
            conn.execute(f"DELETE FROM organization_embeddings WHERE org_id IN ({placeholders})", ids_to_delete)

        # Delete from main table
        cursor = conn.execute("DELETE FROM organizations WHERE source = ?", (source,))
        deleted = cursor.rowcount

        conn.commit()

        logger.info(f"Deleted {deleted} records from source '{source}'")
        return deleted

    def migrate_from_legacy_schema(self) -> dict[str, str]:
        """
        Migrate database from legacy schema (companies/company_embeddings tables)
        to new schema (organizations/organization_embeddings tables).

        This handles:
        - Renaming 'companies' table to 'organizations'
        - Renaming 'company_embeddings' table to 'organization_embeddings'
        - Renaming 'company_id' column to 'org_id' in embeddings table
        - Updating indexes to use new naming

        Returns:
            Dict of migrations performed (table_name -> action)
        """
        conn = self._connect()
        migrations = {}

        # Check what tables exist
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        existing_tables = {row[0] for row in cursor}

        has_companies = "companies" in existing_tables
        has_organizations = "organizations" in existing_tables
        has_company_embeddings = "company_embeddings" in existing_tables
        has_org_embeddings = "organization_embeddings" in existing_tables

        if not has_companies and not has_company_embeddings:
            if has_organizations and has_org_embeddings:
                logger.info("Database already uses new schema, no migration needed")
                return {}
            else:
                logger.info("No legacy tables found, database will use new schema")
                return {}

        logger.info("Migrating database from legacy schema...")
        conn.execute("BEGIN")

        try:
            # Migrate companies -> organizations
            if has_companies:
                if has_organizations:
                    # Both exist - merge data from companies into organizations
                    logger.info("Merging companies table into organizations...")
                    conn.execute("""
                        INSERT OR IGNORE INTO organizations
                        (name, name_normalized, source, source_id, region, entity_type, record)
                        SELECT name, name_normalized, source, source_id,
                               COALESCE(region, ''), COALESCE(entity_type, 'unknown'), record
                        FROM companies
                    """)
                    conn.execute("DROP TABLE companies")
                    migrations["companies"] = "merged_into_organizations"
                else:
                    # Just rename
                    logger.info("Renaming companies table to organizations...")
                    conn.execute("ALTER TABLE companies RENAME TO organizations")
                    migrations["companies"] = "renamed_to_organizations"

                # Update indexes
                for old_idx in ["idx_companies_name", "idx_companies_name_normalized",
                               "idx_companies_source", "idx_companies_source_id",
                               "idx_companies_region", "idx_companies_entity_type",
                               "idx_companies_name_region_source"]:
                    try:
                        conn.execute(f"DROP INDEX IF EXISTS {old_idx}")
                    except Exception:
                        pass

                # Create new indexes
                conn.execute("CREATE INDEX IF NOT EXISTS idx_orgs_name ON organizations(name)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_orgs_name_normalized ON organizations(name_normalized)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_orgs_source ON organizations(source)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_orgs_source_id ON organizations(source, source_id)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_orgs_region ON organizations(region)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_orgs_entity_type ON organizations(entity_type)")
                conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_orgs_name_region_source ON organizations(name, region, source)")

            # Migrate company_embeddings -> organization_embeddings
            if has_company_embeddings:
                if has_org_embeddings:
                    # Both exist - merge
                    logger.info("Merging company_embeddings into organization_embeddings...")
                    # Get column info to check for company_id vs org_id
                    cursor = conn.execute("PRAGMA table_info(company_embeddings)")
                    cols = {row[1] for row in cursor}
                    id_col = "company_id" if "company_id" in cols else "org_id"

                    conn.execute(f"""
                        INSERT OR IGNORE INTO organization_embeddings (org_id, embedding)
                        SELECT {id_col}, embedding FROM company_embeddings
                    """)
                    conn.execute("DROP TABLE company_embeddings")
                    migrations["company_embeddings"] = "merged_into_organization_embeddings"
                else:
                    # Need to recreate with new column name
                    logger.info("Migrating company_embeddings to organization_embeddings...")

                    # Check if it has company_id or org_id column
                    cursor = conn.execute("PRAGMA table_info(company_embeddings)")
                    cols = {row[1] for row in cursor}
                    id_col = "company_id" if "company_id" in cols else "org_id"

                    # Create new virtual table
                    conn.execute(f"""
                        CREATE VIRTUAL TABLE organization_embeddings USING vec0(
                            org_id INTEGER PRIMARY KEY,
                            embedding float[{self._embedding_dim}]
                        )
                    """)

                    # Copy data
                    conn.execute(f"""
                        INSERT INTO organization_embeddings (org_id, embedding)
                        SELECT {id_col}, embedding FROM company_embeddings
                    """)

                    # Drop old table
                    conn.execute("DROP TABLE company_embeddings")
                    migrations["company_embeddings"] = "renamed_to_organization_embeddings"

            conn.execute("COMMIT")
            logger.info(f"Migration complete: {migrations}")

        except Exception as e:
            conn.execute("ROLLBACK")
            logger.error(f"Migration failed: {e}")
            raise

        # Vacuum to clean up - outside try block since COMMIT already succeeded
        try:
            conn.execute("VACUUM")
        except Exception as e:
            logger.warning(f"VACUUM failed (migration was successful): {e}")

        return migrations

    def get_missing_embedding_count(self) -> int:
        """Get count of organizations without embeddings in organization_embeddings table."""
        conn = self._connect()

        cursor = conn.execute("""
            SELECT COUNT(*) FROM organizations c
            LEFT JOIN organization_embeddings e ON c.id = e.org_id
            WHERE e.org_id IS NULL
        """)
        return cursor.fetchone()[0]

    def get_organizations_without_embeddings(
        self,
        batch_size: int = 1000,
        source: Optional[str] = None,
    ) -> Iterator[tuple[int, str]]:
        """
        Iterate over organizations that don't have embeddings.

        Args:
            batch_size: Number of records per batch
            source: Optional source filter

        Yields:
            Tuples of (org_id, name)
        """
        conn = self._connect()

        last_id = 0
        while True:
            if source:
                cursor = conn.execute("""
                    SELECT c.id, c.name FROM organizations c
                    LEFT JOIN organization_embeddings e ON c.id = e.org_id
                    WHERE e.org_id IS NULL AND c.id > ? AND c.source = ?
                    ORDER BY c.id
                    LIMIT ?
                """, (last_id, source, batch_size))
            else:
                cursor = conn.execute("""
                    SELECT c.id, c.name FROM organizations c
                    LEFT JOIN organization_embeddings e ON c.id = e.org_id
                    WHERE e.org_id IS NULL AND c.id > ?
                    ORDER BY c.id
                    LIMIT ?
                """, (last_id, batch_size))

            rows = cursor.fetchall()
            if not rows:
                break

            for row in rows:
                yield (row[0], row[1])
                last_id = row[0]

    def insert_embeddings_batch(
        self,
        org_ids: list[int],
        embeddings: np.ndarray,
    ) -> int:
        """
        Insert embeddings for existing organizations.

        Args:
            org_ids: List of organization IDs
            embeddings: Matrix of embeddings (N x dim)

        Returns:
            Number of embeddings inserted
        """
        conn = self._connect()
        count = 0

        for org_id, embedding in zip(org_ids, embeddings):
            embedding_blob = embedding.astype(np.float32).tobytes()
            conn.execute("DELETE FROM organization_embeddings WHERE org_id = ?", (org_id,))
            conn.execute("""
                INSERT INTO organization_embeddings (org_id, embedding)
                VALUES (?, ?)
            """, (org_id, embedding_blob))
            count += 1

        conn.commit()
        return count

    def resolve_qid_labels(
        self,
        label_map: dict[str, str],
        batch_size: int = 1000,
    ) -> int:
        """
        Update organization records that have QIDs instead of labels in region field.

        Args:
            label_map: Mapping of QID -> label for resolution
            batch_size: Commit batch size

        Returns:
            Number of records updated
        """
        conn = self._connect()

        # Find records with QIDs in region field (starts with 'Q' followed by digits)
        region_updates = 0
        cursor = conn.execute("""
            SELECT id, region FROM organizations
            WHERE region LIKE 'Q%' AND region GLOB 'Q[0-9]*'
        """)
        rows = cursor.fetchall()

        for row in rows:
            org_id = row["id"]
            qid = row["region"]
            if qid in label_map:
                conn.execute(
                    "UPDATE organizations SET region = ? WHERE id = ?",
                    (label_map[qid], org_id)
                )
                region_updates += 1

                if region_updates % batch_size == 0:
                    conn.commit()
                    logger.info(f"Updated {region_updates} organization region labels...")

        conn.commit()
        logger.info(f"Resolved QID labels: {region_updates} organization regions")
        return region_updates

    def get_unresolved_qids(self) -> set[str]:
        """
        Get all QIDs that still need resolution in the organizations table.

        Returns:
            Set of QIDs (starting with 'Q') found in region field
        """
        conn = self._connect()
        qids: set[str] = set()

        cursor = conn.execute("""
            SELECT DISTINCT region FROM organizations
            WHERE region LIKE 'Q%' AND region GLOB 'Q[0-9]*'
        """)
        for row in cursor:
            qids.add(row["region"])

        return qids


def get_person_database(db_path: Optional[str | Path] = None, embedding_dim: int = 768) -> "PersonDatabase":
    """
    Get a singleton PersonDatabase instance for the given path.

    Args:
        db_path: Path to database file
        embedding_dim: Dimension of embeddings

    Returns:
        Shared PersonDatabase instance
    """
    path_key = str(db_path or DEFAULT_DB_PATH)
    if path_key not in _person_database_instances:
        logger.debug(f"Creating new PersonDatabase instance for {path_key}")
        _person_database_instances[path_key] = PersonDatabase(db_path=db_path, embedding_dim=embedding_dim)
    return _person_database_instances[path_key]


class PersonDatabase:
    """
    SQLite database with sqlite-vec for person vector search.

    Uses hybrid text + vector search:
    1. Text filtering with LIKE to reduce candidates
    2. sqlite-vec for semantic similarity ranking

    Stores people from sources like Wikidata with role/org context.
    """

    def __init__(
        self,
        db_path: Optional[str | Path] = None,
        embedding_dim: int = 768,  # Default for embeddinggemma-300m
    ):
        """
        Initialize the person database.

        Args:
            db_path: Path to database file (creates if not exists)
            embedding_dim: Dimension of embeddings to store
        """
        self._db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self._embedding_dim = embedding_dim
        self._conn: Optional[sqlite3.Connection] = None

    def _ensure_dir(self) -> None:
        """Ensure database directory exists."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

    def _connect(self) -> sqlite3.Connection:
        """Get or create database connection using shared connection pool."""
        if self._conn is not None:
            return self._conn

        self._conn = _get_shared_connection(self._db_path, self._embedding_dim)

        # Create tables (idempotent)
        self._create_tables()

        return self._conn

    def _create_tables(self) -> None:
        """Create database tables including sqlite-vec virtual table."""
        conn = self._conn
        assert conn is not None

        # Check if we need to migrate from old schema (unique on source+source_id only)
        self._migrate_people_schema_if_needed(conn)

        # Main people records table
        # Unique constraint on source+source_id+role+org allows multiple records
        # for the same person with different role/org combinations
        conn.execute("""
            CREATE TABLE IF NOT EXISTS people (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                name_normalized TEXT NOT NULL,
                source TEXT NOT NULL DEFAULT 'wikidata',
                source_id TEXT NOT NULL,
                country TEXT NOT NULL DEFAULT '',
                person_type TEXT NOT NULL DEFAULT 'unknown',
                known_for_role TEXT NOT NULL DEFAULT '',
                known_for_org TEXT NOT NULL DEFAULT '',
                known_for_org_id INTEGER DEFAULT NULL,
                from_date TEXT NOT NULL DEFAULT '',
                to_date TEXT NOT NULL DEFAULT '',
                birth_date TEXT NOT NULL DEFAULT '',
                death_date TEXT NOT NULL DEFAULT '',
                record TEXT NOT NULL,
                UNIQUE(source, source_id, known_for_role, known_for_org),
                FOREIGN KEY (known_for_org_id) REFERENCES organizations(id)
            )
        """)

        # Create indexes on main table
        conn.execute("CREATE INDEX IF NOT EXISTS idx_people_name ON people(name)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_people_name_normalized ON people(name_normalized)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_people_source ON people(source)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_people_source_id ON people(source, source_id, known_for_role, known_for_org)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_people_known_for_org ON people(known_for_org)")

        # Add from_date column if it doesn't exist (migration for existing DBs)
        try:
            conn.execute("ALTER TABLE people ADD COLUMN from_date TEXT NOT NULL DEFAULT ''")
            logger.info("Added from_date column to people table")
        except sqlite3.OperationalError:
            pass  # Column already exists

        # Add to_date column if it doesn't exist (migration for existing DBs)
        try:
            conn.execute("ALTER TABLE people ADD COLUMN to_date TEXT NOT NULL DEFAULT ''")
            logger.info("Added to_date column to people table")
        except sqlite3.OperationalError:
            pass  # Column already exists

        # Add known_for_org_id column if it doesn't exist (migration for existing DBs)
        # This is a foreign key to the organizations table (nullable)
        try:
            conn.execute("ALTER TABLE people ADD COLUMN known_for_org_id INTEGER DEFAULT NULL")
            logger.info("Added known_for_org_id column to people table")
        except sqlite3.OperationalError:
            pass  # Column already exists

        # Create index on known_for_org_id for joins (only if column exists)
        try:
            conn.execute("CREATE INDEX IF NOT EXISTS idx_people_known_for_org_id ON people(known_for_org_id)")
        except sqlite3.OperationalError:
            pass  # Column doesn't exist yet (will be added on next connection)

        # Add birth_date column if it doesn't exist (migration for existing DBs)
        try:
            conn.execute("ALTER TABLE people ADD COLUMN birth_date TEXT NOT NULL DEFAULT ''")
            logger.info("Added birth_date column to people table")
        except sqlite3.OperationalError:
            pass  # Column already exists

        # Add death_date column if it doesn't exist (migration for existing DBs)
        try:
            conn.execute("ALTER TABLE people ADD COLUMN death_date TEXT NOT NULL DEFAULT ''")
            logger.info("Added death_date column to people table")
        except sqlite3.OperationalError:
            pass  # Column already exists

        # Add canon_id column if it doesn't exist (migration for canonicalization)
        try:
            conn.execute("ALTER TABLE people ADD COLUMN canon_id INTEGER DEFAULT NULL")
            logger.info("Added canon_id column to people table")
        except sqlite3.OperationalError:
            pass  # Column already exists

        # Add canon_size column if it doesn't exist (migration for canonicalization)
        try:
            conn.execute("ALTER TABLE people ADD COLUMN canon_size INTEGER DEFAULT 1")
            logger.info("Added canon_size column to people table")
        except sqlite3.OperationalError:
            pass  # Column already exists

        # Create index on canon_id for joins
        try:
            conn.execute("CREATE INDEX IF NOT EXISTS idx_people_canon_id ON people(canon_id)")
        except sqlite3.OperationalError:
            pass  # Column doesn't exist yet

        # Create sqlite-vec virtual table for embeddings
        conn.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS person_embeddings USING vec0(
                person_id INTEGER PRIMARY KEY,
                embedding float[{self._embedding_dim}]
            )
        """)

        # Create QID labels lookup table for Wikidata QID -> label mappings
        conn.execute("""
            CREATE TABLE IF NOT EXISTS qid_labels (
                qid TEXT PRIMARY KEY,
                label TEXT NOT NULL
            )
        """)

        conn.commit()

    def _migrate_people_schema_if_needed(self, conn: sqlite3.Connection) -> None:
        """Migrate people table from old schema if needed."""
        # Check if people table exists
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='people'"
        )
        if not cursor.fetchone():
            return  # Table doesn't exist, no migration needed

        # Check the unique constraint - look at index info
        # Old schema: UNIQUE(source, source_id)
        # New schema: UNIQUE(source, source_id, known_for_role, known_for_org)
        cursor = conn.execute("PRAGMA index_list(people)")
        indexes = cursor.fetchall()

        needs_migration = False
        for idx in indexes:
            idx_name = idx[1]
            if "sqlite_autoindex_people" in idx_name:
                # Check columns in this unique index
                cursor = conn.execute(f"PRAGMA index_info('{idx_name}')")
                cols = [row[2] for row in cursor.fetchall()]
                # Old schema has only 2 columns in unique constraint
                if cols == ["source", "source_id"]:
                    needs_migration = True
                    logger.info("Detected old people schema, migrating to new unique constraint...")
                    break

        if not needs_migration:
            return

        # Migrate: create new table, copy data, drop old, rename new
        logger.info("Migrating people table to new schema with (source, source_id, role, org) unique constraint...")

        conn.execute("""
            CREATE TABLE people_new (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                name_normalized TEXT NOT NULL,
                source TEXT NOT NULL DEFAULT 'wikidata',
                source_id TEXT NOT NULL,
                country TEXT NOT NULL DEFAULT '',
                person_type TEXT NOT NULL DEFAULT 'unknown',
                known_for_role TEXT NOT NULL DEFAULT '',
                known_for_org TEXT NOT NULL DEFAULT '',
                known_for_org_id INTEGER DEFAULT NULL,
                from_date TEXT NOT NULL DEFAULT '',
                to_date TEXT NOT NULL DEFAULT '',
                record TEXT NOT NULL,
                UNIQUE(source, source_id, known_for_role, known_for_org),
                FOREIGN KEY (known_for_org_id) REFERENCES organizations(id)
            )
        """)

        # Copy data (old IDs will change, but embeddings table references them)
        # Note: old table may not have from_date/to_date columns, so use defaults
        conn.execute("""
            INSERT INTO people_new (name, name_normalized, source, source_id, country,
                                    person_type, known_for_role, known_for_org, record)
            SELECT name, name_normalized, source, source_id, country,
                   person_type, known_for_role, known_for_org, record
            FROM people
        """)

        # Drop old table and embeddings (IDs changed, embeddings are invalid)
        conn.execute("DROP TABLE IF EXISTS person_embeddings")
        conn.execute("DROP TABLE people")
        conn.execute("ALTER TABLE people_new RENAME TO people")

        # Drop old index if it exists
        conn.execute("DROP INDEX IF EXISTS idx_people_source_id")

        conn.commit()
        logger.info("Migration complete. Note: person embeddings were cleared and need to be regenerated.")

    def close(self) -> None:
        """Clear connection reference (shared connection remains open)."""
        self._conn = None

    def insert(self, record: PersonRecord, embedding: np.ndarray) -> int:
        """
        Insert a person record with its embedding.

        Args:
            record: Person record to insert
            embedding: Embedding vector for the person name

        Returns:
            Row ID of inserted record
        """
        conn = self._connect()

        # Serialize record
        record_json = json.dumps(record.record)
        name_normalized = _normalize_person_name(record.name)

        cursor = conn.execute("""
            INSERT OR REPLACE INTO people
            (name, name_normalized, source, source_id, country, person_type,
             known_for_role, known_for_org, known_for_org_id, from_date, to_date,
             birth_date, death_date, record)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            record.name,
            name_normalized,
            record.source,
            record.source_id,
            record.country,
            record.person_type.value,
            record.known_for_role,
            record.known_for_org,
            record.known_for_org_id,  # Can be None
            record.from_date or "",
            record.to_date or "",
            record.birth_date or "",
            record.death_date or "",
            record_json,
        ))

        row_id = cursor.lastrowid
        assert row_id is not None

        # Insert embedding into vec table (delete first since sqlite-vec doesn't support REPLACE)
        embedding_blob = embedding.astype(np.float32).tobytes()
        conn.execute("DELETE FROM person_embeddings WHERE person_id = ?", (row_id,))
        conn.execute("""
            INSERT INTO person_embeddings (person_id, embedding)
            VALUES (?, ?)
        """, (row_id, embedding_blob))

        conn.commit()
        return row_id

    def insert_batch(
        self,
        records: list[PersonRecord],
        embeddings: np.ndarray,
        batch_size: int = 1000,
    ) -> int:
        """
        Insert multiple person records with embeddings.

        Args:
            records: List of person records
            embeddings: Matrix of embeddings (N x dim)
            batch_size: Commit batch size

        Returns:
            Number of records inserted
        """
        conn = self._connect()
        count = 0

        for record, embedding in zip(records, embeddings):
            record_json = json.dumps(record.record)
            name_normalized = _normalize_person_name(record.name)

            cursor = conn.execute("""
                INSERT OR REPLACE INTO people
                (name, name_normalized, source, source_id, country, person_type,
                 known_for_role, known_for_org, known_for_org_id, from_date, to_date,
                 birth_date, death_date, record)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.name,
                name_normalized,
                record.source,
                record.source_id,
                record.country,
                record.person_type.value,
                record.known_for_role,
                record.known_for_org,
                record.known_for_org_id,  # Can be None
                record.from_date or "",
                record.to_date or "",
                record.birth_date or "",
                record.death_date or "",
                record_json,
            ))

            row_id = cursor.lastrowid
            assert row_id is not None

            # Insert embedding (delete first since sqlite-vec doesn't support REPLACE)
            embedding_blob = embedding.astype(np.float32).tobytes()
            conn.execute("DELETE FROM person_embeddings WHERE person_id = ?", (row_id,))
            conn.execute("""
                INSERT INTO person_embeddings (person_id, embedding)
                VALUES (?, ?)
            """, (row_id, embedding_blob))

            count += 1

            if count % batch_size == 0:
                conn.commit()
                logger.info(f"Inserted {count} person records...")

        conn.commit()
        return count

    def update_dates(self, source: str, source_id: str, from_date: Optional[str], to_date: Optional[str]) -> bool:
        """
        Update the from_date and to_date for a person record.

        Args:
            source: Data source (e.g., 'wikidata')
            source_id: Source identifier (e.g., QID)
            from_date: Start date in ISO format or None
            to_date: End date in ISO format or None

        Returns:
            True if record was updated, False if not found
        """
        conn = self._connect()

        cursor = conn.execute("""
            UPDATE people SET from_date = ?, to_date = ?
            WHERE source = ? AND source_id = ?
        """, (from_date or "", to_date or "", source, source_id))

        conn.commit()
        return cursor.rowcount > 0

    def update_role_org(
        self,
        source: str,
        source_id: str,
        known_for_role: str,
        known_for_org: str,
        known_for_org_id: Optional[int],
        new_embedding: np.ndarray,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
    ) -> bool:
        """
        Update the role/org/dates data for a person record and re-embed.

        Args:
            source: Data source (e.g., 'wikidata')
            source_id: Source identifier (e.g., QID)
            known_for_role: Role/position title
            known_for_org: Organization name
            known_for_org_id: Organization internal ID (FK) or None
            new_embedding: New embedding vector based on updated data
            from_date: Start date in ISO format or None
            to_date: End date in ISO format or None

        Returns:
            True if record was updated, False if not found
        """
        conn = self._connect()

        # First get the person's internal ID
        row = conn.execute(
            "SELECT id FROM people WHERE source = ? AND source_id = ?",
            (source, source_id)
        ).fetchone()

        if not row:
            return False

        person_id = row[0]

        # Update the person record (including dates)
        conn.execute("""
            UPDATE people SET
                known_for_role = ?, known_for_org = ?, known_for_org_id = ?,
                from_date = COALESCE(?, from_date, ''),
                to_date = COALESCE(?, to_date, '')
            WHERE id = ?
        """, (known_for_role, known_for_org, known_for_org_id, from_date, to_date, person_id))

        # Update the embedding
        embedding_bytes = new_embedding.astype(np.float32).tobytes()
        conn.execute("""
            UPDATE people_vec SET embedding = ?
            WHERE rowid = ?
        """, (embedding_bytes, person_id))

        conn.commit()
        return True

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 20,
        query_text: Optional[str] = None,
        max_text_candidates: int = 5000,
    ) -> list[tuple[PersonRecord, float]]:
        """
        Search for similar people using hybrid text + vector search.

        Two-stage approach:
        1. If query_text provided, use SQL LIKE to find candidates containing search terms
        2. Use sqlite-vec for vector similarity ranking on filtered candidates

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            query_text: Optional query text for text-based pre-filtering
            max_text_candidates: Max candidates to keep after text filtering

        Returns:
            List of (PersonRecord, similarity_score) tuples
        """
        start = time.time()
        self._connect()

        # Normalize query embedding
        query_norm = np.linalg.norm(query_embedding)
        if query_norm == 0:
            return []
        query_normalized = query_embedding / query_norm
        query_blob = query_normalized.astype(np.float32).tobytes()

        # Stage 1: Text-based pre-filtering (if query_text provided)
        candidate_ids: Optional[set[int]] = None
        if query_text:
            query_normalized_text = _normalize_person_name(query_text)
            if query_normalized_text:
                candidate_ids = self._text_filter_candidates(
                    query_normalized_text,
                    max_candidates=max_text_candidates,
                )
                logger.info(f"Text filter: {len(candidate_ids)} candidates for '{query_text}'")

        # Stage 2: Vector search
        if candidate_ids is not None and len(candidate_ids) == 0:
            # No text matches, return empty
            return []

        if candidate_ids is not None:
            # Search within text-filtered candidates
            results = self._vector_search_filtered(
                query_blob, candidate_ids, top_k
            )
        else:
            # Full vector search
            results = self._vector_search_full(query_blob, top_k)

        elapsed = time.time() - start
        logger.debug(f"Person search took {elapsed:.3f}s (results={len(results)})")
        return results

    def _text_filter_candidates(
        self,
        query_normalized: str,
        max_candidates: int,
    ) -> set[int]:
        """
        Filter candidates using SQL LIKE for fast text matching.

        Uses `name_normalized` column for consistent matching.
        """
        conn = self._conn
        assert conn is not None

        # Extract search terms from the normalized query
        search_terms = _extract_search_terms(query_normalized)
        if not search_terms:
            return set()

        logger.debug(f"Person text filter search terms: {search_terms}")

        # Build OR clause for LIKE matching on any term
        like_clauses = []
        params: list = []
        for term in search_terms:
            like_clauses.append("name_normalized LIKE ?")
            params.append(f"%{term}%")

        where_clause = " OR ".join(like_clauses)

        query = f"""
            SELECT id FROM people
            WHERE {where_clause}
            LIMIT ?
        """

        params.append(max_candidates)

        cursor = conn.execute(query, params)
        return set(row["id"] for row in cursor)

    def _vector_search_filtered(
        self,
        query_blob: bytes,
        candidate_ids: set[int],
        top_k: int,
    ) -> list[tuple[PersonRecord, float]]:
        """Vector search within a filtered set of candidates."""
        conn = self._conn
        assert conn is not None

        if not candidate_ids:
            return []

        # Build IN clause for candidate IDs
        placeholders = ",".join("?" * len(candidate_ids))

        query = f"""
            SELECT
                e.person_id,
                vec_distance_cosine(e.embedding, ?) as distance
            FROM person_embeddings e
            WHERE e.person_id IN ({placeholders})
            ORDER BY distance
            LIMIT ?
        """

        cursor = conn.execute(query, [query_blob] + list(candidate_ids) + [top_k])

        results = []
        for row in cursor:
            person_id = row["person_id"]
            distance = row["distance"]
            # Convert cosine distance to similarity (1 - distance)
            similarity = 1.0 - distance

            # Fetch full record
            record = self._get_record_by_id(person_id)
            if record:
                results.append((record, similarity))

        return results

    def _vector_search_full(
        self,
        query_blob: bytes,
        top_k: int,
    ) -> list[tuple[PersonRecord, float]]:
        """Full vector search without text pre-filtering."""
        conn = self._conn
        assert conn is not None

        query = """
            SELECT
                person_id,
                vec_distance_cosine(embedding, ?) as distance
            FROM person_embeddings
            ORDER BY distance
            LIMIT ?
        """
        cursor = conn.execute(query, (query_blob, top_k))

        results = []
        for row in cursor:
            person_id = row["person_id"]
            distance = row["distance"]
            similarity = 1.0 - distance

            record = self._get_record_by_id(person_id)
            if record:
                results.append((record, similarity))

        return results

    def _get_record_by_id(self, person_id: int) -> Optional[PersonRecord]:
        """Get a person record by ID."""
        conn = self._conn
        assert conn is not None

        cursor = conn.execute("""
            SELECT name, source, source_id, country, person_type, known_for_role, known_for_org, known_for_org_id, birth_date, death_date, record
            FROM people WHERE id = ?
        """, (person_id,))

        row = cursor.fetchone()
        if row:
            return PersonRecord(
                name=row["name"],
                source=row["source"],
                source_id=row["source_id"],
                country=row["country"] or "",
                person_type=PersonType(row["person_type"]) if row["person_type"] else PersonType.UNKNOWN,
                known_for_role=row["known_for_role"] or "",
                known_for_org=row["known_for_org"] or "",
                known_for_org_id=row["known_for_org_id"],  # Can be None
                birth_date=row["birth_date"] or "",
                death_date=row["death_date"] or "",
                record=json.loads(row["record"]),
            )
        return None

    def get_by_source_id(self, source: str, source_id: str) -> Optional[PersonRecord]:
        """Get a person record by source and source_id."""
        conn = self._connect()

        cursor = conn.execute("""
            SELECT name, source, source_id, country, person_type, known_for_role, known_for_org, known_for_org_id, birth_date, death_date, record
            FROM people
            WHERE source = ? AND source_id = ?
        """, (source, source_id))

        row = cursor.fetchone()
        if row:
            return PersonRecord(
                name=row["name"],
                source=row["source"],
                source_id=row["source_id"],
                country=row["country"] or "",
                person_type=PersonType(row["person_type"]) if row["person_type"] else PersonType.UNKNOWN,
                known_for_role=row["known_for_role"] or "",
                known_for_org=row["known_for_org"] or "",
                known_for_org_id=row["known_for_org_id"],  # Can be None
                birth_date=row["birth_date"] or "",
                death_date=row["death_date"] or "",
                record=json.loads(row["record"]),
            )
        return None

    def get_stats(self) -> dict:
        """Get database statistics for people table."""
        conn = self._connect()

        # Total count
        cursor = conn.execute("SELECT COUNT(*) FROM people")
        total = cursor.fetchone()[0]

        # Count by person_type
        cursor = conn.execute("SELECT person_type, COUNT(*) as cnt FROM people GROUP BY person_type")
        by_type = {row["person_type"]: row["cnt"] for row in cursor}

        # Count by source
        cursor = conn.execute("SELECT source, COUNT(*) as cnt FROM people GROUP BY source")
        by_source = {row["source"]: row["cnt"] for row in cursor}

        return {
            "total_records": total,
            "by_type": by_type,
            "by_source": by_source,
        }

    def get_all_source_ids(self, source: Optional[str] = None) -> set[str]:
        """
        Get all source_ids from the people table.

        Useful for resume operations to skip already-imported records.

        Args:
            source: Optional source filter (e.g., "wikidata")

        Returns:
            Set of source_id strings (e.g., Q codes for Wikidata)
        """
        conn = self._connect()

        if source:
            cursor = conn.execute(
                "SELECT DISTINCT source_id FROM people WHERE source = ?",
                (source,)
            )
        else:
            cursor = conn.execute("SELECT DISTINCT source_id FROM people")

        return {row[0] for row in cursor}

    def iter_records(self, source: Optional[str] = None) -> Iterator[PersonRecord]:
        """Iterate over all person records, optionally filtered by source."""
        conn = self._connect()

        if source:
            cursor = conn.execute("""
                SELECT name, source, source_id, country, person_type, known_for_role, known_for_org, known_for_org_id, birth_date, death_date, record
                FROM people
                WHERE source = ?
            """, (source,))
        else:
            cursor = conn.execute("""
                SELECT name, source, source_id, country, person_type, known_for_role, known_for_org, known_for_org_id, birth_date, death_date, record
                FROM people
            """)

        for row in cursor:
            yield PersonRecord(
                name=row["name"],
                source=row["source"],
                source_id=row["source_id"],
                country=row["country"] or "",
                person_type=PersonType(row["person_type"]) if row["person_type"] else PersonType.UNKNOWN,
                known_for_role=row["known_for_role"] or "",
                known_for_org=row["known_for_org"] or "",
                known_for_org_id=row["known_for_org_id"],  # Can be None
                birth_date=row["birth_date"] or "",
                death_date=row["death_date"] or "",
                record=json.loads(row["record"]),
            )

    def resolve_qid_labels(
        self,
        label_map: dict[str, str],
        batch_size: int = 1000,
    ) -> tuple[int, int]:
        """
        Update records that have QIDs instead of labels.

        This is called after dump import to resolve any QIDs that were
        stored because labels weren't available in the cache at import time.

        If resolving would create a duplicate of an existing record with
        resolved labels, the QID version is deleted instead.

        Args:
            label_map: Mapping of QID -> label for resolution
            batch_size: Commit batch size

        Returns:
            Tuple of (updates, deletes)
        """
        conn = self._connect()

        # Find all records with QIDs in any field (role or org - these are in unique constraint)
        # Country is not part of unique constraint so can be updated directly
        cursor = conn.execute("""
            SELECT id, source, source_id, country, known_for_role, known_for_org
            FROM people
            WHERE (country LIKE 'Q%' AND country GLOB 'Q[0-9]*')
               OR (known_for_role LIKE 'Q%' AND known_for_role GLOB 'Q[0-9]*')
               OR (known_for_org LIKE 'Q%' AND known_for_org GLOB 'Q[0-9]*')
        """)
        rows = cursor.fetchall()

        updates = 0
        deletes = 0

        for row in rows:
            person_id = row["id"]
            source = row["source"]
            source_id = row["source_id"]
            country = row["country"]
            role = row["known_for_role"]
            org = row["known_for_org"]

            # Resolve QIDs to labels
            new_country = label_map.get(country, country) if country.startswith("Q") and country[1:].isdigit() else country
            new_role = label_map.get(role, role) if role.startswith("Q") and role[1:].isdigit() else role
            new_org = label_map.get(org, org) if org.startswith("Q") and org[1:].isdigit() else org

            # Skip if nothing changed
            if new_country == country and new_role == role and new_org == org:
                continue

            # Check if resolved values would duplicate an existing record
            # (unique constraint is on source, source_id, known_for_role, known_for_org)
            if new_role != role or new_org != org:
                cursor2 = conn.execute("""
                    SELECT id FROM people
                    WHERE source = ? AND source_id = ? AND known_for_role = ? AND known_for_org = ?
                    AND id != ?
                """, (source, source_id, new_role, new_org, person_id))
                existing = cursor2.fetchone()

                if existing:
                    # Duplicate would exist - delete the QID version
                    conn.execute("DELETE FROM people WHERE id = ?", (person_id,))
                    conn.execute("DELETE FROM person_embeddings WHERE person_id = ?", (person_id,))
                    deletes += 1
                    logger.debug(f"Deleted duplicate QID record {person_id} (source_id={source_id})")
                    continue

            # No duplicate - update in place
            conn.execute("""
                UPDATE people SET country = ?, known_for_role = ?, known_for_org = ?
                WHERE id = ?
            """, (new_country, new_role, new_org, person_id))
            updates += 1

            if (updates + deletes) % batch_size == 0:
                conn.commit()
                logger.info(f"Resolved QID labels: {updates} updates, {deletes} deletes...")

        conn.commit()
        logger.info(f"Resolved QID labels: {updates} updates, {deletes} deletes")
        return updates, deletes

    def get_unresolved_qids(self) -> set[str]:
        """
        Get all QIDs that still need resolution in the database.

        Returns:
            Set of QIDs (starting with 'Q') found in country, role, or org fields
        """
        conn = self._connect()
        qids: set[str] = set()

        # Get QIDs from country field
        cursor = conn.execute("""
            SELECT DISTINCT country FROM people
            WHERE country LIKE 'Q%' AND country GLOB 'Q[0-9]*'
        """)
        for row in cursor:
            qids.add(row["country"])

        # Get QIDs from known_for_role field
        cursor = conn.execute("""
            SELECT DISTINCT known_for_role FROM people
            WHERE known_for_role LIKE 'Q%' AND known_for_role GLOB 'Q[0-9]*'
        """)
        for row in cursor:
            qids.add(row["known_for_role"])

        # Get QIDs from known_for_org field
        cursor = conn.execute("""
            SELECT DISTINCT known_for_org FROM people
            WHERE known_for_org LIKE 'Q%' AND known_for_org GLOB 'Q[0-9]*'
        """)
        for row in cursor:
            qids.add(row["known_for_org"])

        return qids

    def insert_qid_labels(
        self,
        label_map: dict[str, str],
        batch_size: int = 1000,
    ) -> int:
        """
        Insert QID -> label mappings into the lookup table.

        Args:
            label_map: Mapping of QID -> label
            batch_size: Commit batch size

        Returns:
            Number of labels inserted/updated
        """
        conn = self._connect()
        count = 0

        for qid, label in label_map.items():
            conn.execute(
                "INSERT OR REPLACE INTO qid_labels (qid, label) VALUES (?, ?)",
                (qid, label)
            )
            count += 1

            if count % batch_size == 0:
                conn.commit()
                logger.debug(f"Inserted {count} QID labels...")

        conn.commit()
        logger.info(f"Inserted {count} QID labels into lookup table")
        return count

    def get_qid_label(self, qid: str) -> Optional[str]:
        """
        Get the label for a QID from the lookup table.

        Args:
            qid: Wikidata QID (e.g., 'Q30')

        Returns:
            Label string or None if not found
        """
        conn = self._connect()
        cursor = conn.execute(
            "SELECT label FROM qid_labels WHERE qid = ?",
            (qid,)
        )
        row = cursor.fetchone()
        return row["label"] if row else None

    def get_all_qid_labels(self) -> dict[str, str]:
        """
        Get all QID -> label mappings from the lookup table.

        Returns:
            Dict mapping QID -> label
        """
        conn = self._connect()
        cursor = conn.execute("SELECT qid, label FROM qid_labels")
        return {row["qid"]: row["label"] for row in cursor}

    def get_qid_labels_count(self) -> int:
        """Get the number of QID labels in the lookup table."""
        conn = self._connect()
        cursor = conn.execute("SELECT COUNT(*) FROM qid_labels")
        return cursor.fetchone()[0]

    def canonicalize(self, batch_size: int = 10000) -> dict[str, int]:
        """
        Canonicalize person records by linking equivalent entries across sources.

        Uses a multi-phase approach:
        1. Match by normalized name + same organization (org canonical group)
        2. Match by normalized name + overlapping date ranges

        Source priority (lower = more authoritative):
        - wikidata: 1 (curated, has Q codes)
        - sec_edgar: 2 (US insider filings)
        - companies_house: 3 (UK officers)

        Args:
            batch_size: Number of records to process before committing

        Returns:
            Stats dict with counts for each matching type
        """
        conn = self._connect()
        stats = {
            "total_records": 0,
            "matched_by_org": 0,
            "matched_by_date": 0,
            "canonical_groups": 0,
            "records_in_groups": 0,
        }

        logger.info("Phase 1: Building person index...")

        # Load all people with their normalized names and org info
        cursor = conn.execute("""
            SELECT id, name, name_normalized, source, source_id,
                   known_for_org, known_for_org_id, from_date, to_date
            FROM people
        """)

        people: list[dict] = []
        for row in cursor:
            people.append({
                "id": row["id"],
                "name": row["name"],
                "name_normalized": row["name_normalized"],
                "source": row["source"],
                "source_id": row["source_id"],
                "known_for_org": row["known_for_org"],
                "known_for_org_id": row["known_for_org_id"],
                "from_date": row["from_date"],
                "to_date": row["to_date"],
            })

        stats["total_records"] = len(people)
        logger.info(f"Loaded {len(people)} person records")

        if len(people) == 0:
            return stats

        # Initialize Union-Find
        person_ids = [p["id"] for p in people]
        uf = UnionFind(person_ids)

        # Build indexes for efficient matching
        # Index by normalized name
        name_to_people: dict[str, list[dict]] = {}
        for p in people:
            name_norm = p["name_normalized"]
            name_to_people.setdefault(name_norm, []).append(p)

        logger.info("Phase 2: Matching by normalized name + organization...")

        # Match people with same normalized name and same organization
        for name_norm, same_name in name_to_people.items():
            if len(same_name) < 2:
                continue

            # Group by organization (using known_for_org_id if available, else known_for_org)
            org_groups: dict[str, list[dict]] = {}
            for p in same_name:
                org_key = str(p["known_for_org_id"]) if p["known_for_org_id"] else p["known_for_org"]
                if org_key:  # Only group if they have an org
                    org_groups.setdefault(org_key, []).append(p)

            # Union people with same name + same org
            for org_key, org_people in org_groups.items():
                if len(org_people) >= 2:
                    first_id = org_people[0]["id"]
                    for p in org_people[1:]:
                        uf.union(first_id, p["id"])
                        stats["matched_by_org"] += 1

        logger.info(f"Phase 2 complete: {stats['matched_by_org']} matches by org")

        logger.info("Phase 3: Matching by normalized name + overlapping dates...")

        # Match people with same normalized name and overlapping date ranges
        for name_norm, same_name in name_to_people.items():
            if len(same_name) < 2:
                continue

            # Skip if already all unified
            roots = set(uf.find(p["id"]) for p in same_name)
            if len(roots) == 1:
                continue

            # Check for overlapping date ranges
            for i, p1 in enumerate(same_name):
                for p2 in same_name[i+1:]:
                    # Skip if already in same group
                    if uf.find(p1["id"]) == uf.find(p2["id"]):
                        continue

                    # Check date overlap (if both have dates)
                    if p1["from_date"] and p2["from_date"]:
                        # Simple overlap check: if either from_date is before other's to_date
                        p1_from = p1["from_date"]
                        p1_to = p1["to_date"] or "9999-12-31"
                        p2_from = p2["from_date"]
                        p2_to = p2["to_date"] or "9999-12-31"

                        # Overlap if: p1_from <= p2_to AND p2_from <= p1_to
                        if p1_from <= p2_to and p2_from <= p1_to:
                            uf.union(p1["id"], p2["id"])
                            stats["matched_by_date"] += 1

        logger.info(f"Phase 3 complete: {stats['matched_by_date']} matches by date")

        logger.info("Phase 4: Applying canonical updates...")

        # Get all groups and select canonical record for each
        groups = uf.groups()

        # Build id -> source mapping
        id_to_source = {p["id"]: p["source"] for p in people}

        batch_updates: list[tuple[int, int, int]] = []  # (person_id, canon_id, canon_size)

        for _root, group_ids in groups.items():
            group_size = len(group_ids)

            if group_size == 1:
                # Single record is its own canonical
                person_id = group_ids[0]
                batch_updates.append((person_id, person_id, 1))
            else:
                # Multiple records - pick highest priority source as canonical
                # Sort by source priority, then by id (for stability)
                sorted_ids = sorted(
                    group_ids,
                    key=lambda pid: (PERSON_SOURCE_PRIORITY.get(id_to_source[pid], 99), pid)
                )
                canon_id = sorted_ids[0]
                stats["canonical_groups"] += 1
                stats["records_in_groups"] += group_size

                for person_id in group_ids:
                    batch_updates.append((person_id, canon_id, group_size if person_id == canon_id else 1))

            # Commit in batches
            if len(batch_updates) >= batch_size:
                self._apply_person_canon_updates(batch_updates)
                conn.commit()
                logger.info(f"Applied {len(batch_updates)} canon updates...")
                batch_updates = []

        # Final batch
        if batch_updates:
            self._apply_person_canon_updates(batch_updates)
            conn.commit()

        logger.info(f"Canonicalization complete: {stats['canonical_groups']} groups, "
                   f"{stats['records_in_groups']} records in multi-record groups")

        return stats

    def _apply_person_canon_updates(self, updates: list[tuple[int, int, int]]) -> None:
        """Apply batch of canon updates: (person_id, canon_id, canon_size)."""
        conn = self._conn
        assert conn is not None

        for person_id, canon_id, canon_size in updates:
            conn.execute(
                "UPDATE people SET canon_id = ?, canon_size = ? WHERE id = ?",
                (canon_id, canon_size, person_id)
            )

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
import sqlite_vec

from .models import CompanyRecord, DatabaseStats, EntityType, PersonRecord, PersonType

logger = logging.getLogger(__name__)

# Default database location
DEFAULT_DB_PATH = Path.home() / ".cache" / "corp-extractor" / "entities.db"

# Module-level singleton for OrganizationDatabase to prevent multiple loads
_database_instances: dict[str, "OrganizationDatabase"] = {}

# Module-level singleton for PersonDatabase
_person_database_instances: dict[str, "PersonDatabase"] = {}

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
        """Get or create database connection with sqlite-vec loaded."""
        if self._conn is not None:
            return self._conn

        self._ensure_dir()
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.row_factory = sqlite3.Row

        # Load sqlite-vec extension
        self._conn.enable_load_extension(True)
        sqlite_vec.load(self._conn)
        self._conn.enable_load_extension(False)

        # Create tables
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

        # Create indexes on main table
        conn.execute("CREATE INDEX IF NOT EXISTS idx_orgs_name ON organizations(name)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_orgs_name_normalized ON organizations(name_normalized)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_orgs_source ON organizations(source)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_orgs_source_id ON organizations(source, source_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_orgs_region ON organizations(region)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_orgs_entity_type ON organizations(entity_type)")
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_orgs_name_region_source ON organizations(name, region, source)")

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
        """Close database connection."""
        if self._conn:
            self._conn.close()
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
            (name, name_normalized, source, source_id, region, entity_type, record)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            record.name,
            name_normalized,
            record.source,
            record.source_id,
            record.region,
            record.entity_type.value,
            record_json,
        ))

        row_id = cursor.lastrowid
        assert row_id is not None

        # Insert embedding into vec table
        # sqlite-vec expects the embedding as a blob
        embedding_blob = embedding.astype(np.float32).tobytes()
        conn.execute("""
            INSERT OR REPLACE INTO organization_embeddings (org_id, embedding)
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
                (name, name_normalized, source, source_id, region, entity_type, record)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                record.name,
                name_normalized,
                record.source,
                record.source_id,
                record.region,
                record.entity_type.value,
                record_json,
            ))

            row_id = cursor.lastrowid
            assert row_id is not None

            # Insert embedding
            embedding_blob = embedding.astype(np.float32).tobytes()
            conn.execute("""
                INSERT OR REPLACE INTO organization_embeddings (org_id, embedding)
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
    ) -> list[tuple[CompanyRecord, float]]:
        """
        Search for similar organizations using hybrid text + vector search.

        Two-stage approach:
        1. If query_text provided, use SQL LIKE to find candidates containing search terms
        2. Use sqlite-vec for vector similarity ranking on filtered candidates

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            source_filter: Optional filter by source (gleif, sec_edgar, etc.)
            query_text: Optional query text for text-based pre-filtering
            max_text_candidates: Max candidates to keep after text filtering

        Returns:
            List of (CompanyRecord, similarity_score) tuples
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
            query_normalized_text = _normalize_name(query_text)
            if query_normalized_text:
                candidate_ids = self._text_filter_candidates(
                    query_normalized_text,
                    max_candidates=max_text_candidates,
                    source_filter=source_filter,
                )
                logger.info(f"Text filter: {len(candidate_ids)} candidates for '{query_text}'")

        # Stage 2: Vector search
        if candidate_ids is not None and len(candidate_ids) == 0:
            # No text matches, return empty
            return []

        if candidate_ids is not None:
            # Search within text-filtered candidates
            results = self._vector_search_filtered(
                query_blob, candidate_ids, top_k, source_filter
            )
        else:
            # Full vector search
            results = self._vector_search_full(query_blob, top_k, source_filter)

        elapsed = time.time() - start
        logger.debug(f"Hybrid search took {elapsed:.3f}s (results={len(results)})")
        return results

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
        """Get an organization record by ID."""
        conn = self._conn
        assert conn is not None

        cursor = conn.execute("""
            SELECT name, source, source_id, region, entity_type, record
            FROM organizations WHERE id = ?
        """, (org_id,))

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
            conn.execute("""
                INSERT OR REPLACE INTO organization_embeddings (org_id, embedding)
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

            # Vacuum to clean up
            conn.execute("VACUUM")
            logger.info(f"Migration complete: {migrations}")

            return migrations

        except Exception as e:
            conn.execute("ROLLBACK")
            logger.error(f"Migration failed: {e}")
            raise

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
            conn.execute("""
                INSERT OR REPLACE INTO organization_embeddings (org_id, embedding)
                VALUES (?, ?)
            """, (org_id, embedding_blob))
            count += 1

        conn.commit()
        return count


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
        """Get or create database connection with sqlite-vec loaded."""
        if self._conn is not None:
            return self._conn

        self._ensure_dir()
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.row_factory = sqlite3.Row

        # Load sqlite-vec extension
        self._conn.enable_load_extension(True)
        sqlite_vec.load(self._conn)
        self._conn.enable_load_extension(False)

        # Create tables
        self._create_tables()

        return self._conn

    def _create_tables(self) -> None:
        """Create database tables including sqlite-vec virtual table."""
        conn = self._conn
        assert conn is not None

        # Main people records table
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
                record TEXT NOT NULL,
                UNIQUE(source, source_id)
            )
        """)

        # Create indexes on main table
        conn.execute("CREATE INDEX IF NOT EXISTS idx_people_name ON people(name)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_people_name_normalized ON people(name_normalized)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_people_source ON people(source)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_people_source_id ON people(source, source_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_people_known_for_org ON people(known_for_org)")

        # Create sqlite-vec virtual table for embeddings
        conn.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS person_embeddings USING vec0(
                person_id INTEGER PRIMARY KEY,
                embedding float[{self._embedding_dim}]
            )
        """)

        conn.commit()

    def close(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()
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
            (name, name_normalized, source, source_id, country, person_type, known_for_role, known_for_org, record)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            record.name,
            name_normalized,
            record.source,
            record.source_id,
            record.country,
            record.person_type.value,
            record.known_for_role,
            record.known_for_org,
            record_json,
        ))

        row_id = cursor.lastrowid
        assert row_id is not None

        # Insert embedding into vec table
        embedding_blob = embedding.astype(np.float32).tobytes()
        conn.execute("""
            INSERT OR REPLACE INTO person_embeddings (person_id, embedding)
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
                (name, name_normalized, source, source_id, country, person_type, known_for_role, known_for_org, record)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.name,
                name_normalized,
                record.source,
                record.source_id,
                record.country,
                record.person_type.value,
                record.known_for_role,
                record.known_for_org,
                record_json,
            ))

            row_id = cursor.lastrowid
            assert row_id is not None

            # Insert embedding
            embedding_blob = embedding.astype(np.float32).tobytes()
            conn.execute("""
                INSERT OR REPLACE INTO person_embeddings (person_id, embedding)
                VALUES (?, ?)
            """, (row_id, embedding_blob))

            count += 1

            if count % batch_size == 0:
                conn.commit()
                logger.info(f"Inserted {count} person records...")

        conn.commit()
        return count

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
            SELECT name, source, source_id, country, person_type, known_for_role, known_for_org, record
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
                record=json.loads(row["record"]),
            )
        return None

    def get_by_source_id(self, source: str, source_id: str) -> Optional[PersonRecord]:
        """Get a person record by source and source_id."""
        conn = self._connect()

        cursor = conn.execute("""
            SELECT name, source, source_id, country, person_type, known_for_role, known_for_org, record
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

    def iter_records(self, source: Optional[str] = None) -> Iterator[PersonRecord]:
        """Iterate over all person records, optionally filtered by source."""
        conn = self._connect()

        if source:
            cursor = conn.execute("""
                SELECT name, source, source_id, country, person_type, known_for_role, known_for_org, record
                FROM people
                WHERE source = ?
            """, (source,))
        else:
            cursor = conn.execute("""
                SELECT name, source, source_id, country, person_type, known_for_role, known_for_org, record
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
                record=json.loads(row["record"]),
            )

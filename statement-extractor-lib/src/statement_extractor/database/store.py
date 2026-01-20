"""
Company database with in-memory vector search.

Loads embeddings into memory for fast numpy-based similarity search.
SQLite is used only for record storage and retrieval.
"""

import json
import logging
import sqlite3
import struct
import time
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

from .models import CompanyRecord, DatabaseStats

logger = logging.getLogger(__name__)

# Default database location
DEFAULT_DB_PATH = Path.home() / ".cache" / "corp-extractor" / "companies.db"


def _serialize_embedding(embedding: np.ndarray) -> bytes:
    """Serialize numpy array to bytes for storage."""
    return struct.pack(f"{len(embedding)}f", *embedding.tolist())


def _deserialize_embedding(data: bytes, dim: int) -> np.ndarray:
    """Deserialize bytes to numpy array."""
    return np.array(struct.unpack(f"{dim}f", data), dtype=np.float32)


class CompanyDatabase:
    """
    SQLite database with in-memory vector search for companies.

    Loads all embeddings into memory for fast numpy-based similarity search.
    SQLite is used only for record storage and retrieval by ID.
    """

    def __init__(
        self,
        db_path: Optional[str | Path] = None,
        embedding_dim: int = 768,  # Default for embeddinggemma-300m
    ):
        """
        Initialize the company database.

        Args:
            db_path: Path to database file (creates if not exists)
            embedding_dim: Dimension of embeddings to store
        """
        self._db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self._embedding_dim = embedding_dim
        self._conn: Optional[sqlite3.Connection] = None

        # In-memory index for fast search
        self._index_loaded = False
        self._embeddings: Optional[np.ndarray] = None  # (N, dim) matrix
        self._row_ids: Optional[np.ndarray] = None  # (N,) array of SQLite row IDs

    def _ensure_dir(self) -> None:
        """Ensure database directory exists."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

    def _connect(self) -> sqlite3.Connection:
        """Get or create database connection."""
        if self._conn is not None:
            return self._conn

        self._ensure_dir()
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.row_factory = sqlite3.Row

        # Create tables
        self._create_tables()

        return self._conn

    def _create_tables(self) -> None:
        """Create database tables."""
        conn = self._conn
        assert conn is not None

        # Main company records table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS companies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                embedding_name TEXT NOT NULL,
                legal_name TEXT NOT NULL,
                source TEXT NOT NULL,
                source_id TEXT NOT NULL,
                region TEXT NOT NULL DEFAULT '',
                record TEXT NOT NULL,
                embedding BLOB,
                UNIQUE(source, source_id)
            )
        """)

        # Add region column if it doesn't exist (migration for existing DBs)
        try:
            conn.execute("ALTER TABLE companies ADD COLUMN region TEXT NOT NULL DEFAULT ''")
            logger.info("Added region column to companies table")
        except sqlite3.OperationalError:
            pass  # Column already exists

        # Create indexes
        conn.execute("CREATE INDEX IF NOT EXISTS idx_companies_name ON companies(name)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_companies_source ON companies(source)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_companies_source_id ON companies(source, source_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_companies_region ON companies(region)")
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_companies_name_region_source ON companies(name, region, source)")

        conn.commit()

    def _load_index(self) -> None:
        """Load all embeddings into memory for fast search."""
        if self._index_loaded:
            return

        conn = self._connect()
        start = time.time()

        logger.info("Loading embedding index into memory...")

        # Count records first
        cursor = conn.execute("SELECT COUNT(*) FROM companies WHERE embedding IS NOT NULL")
        count = cursor.fetchone()[0]

        if count == 0:
            logger.warning("No embeddings found in database")
            self._embeddings = np.zeros((0, self._embedding_dim), dtype=np.float32)
            self._row_ids = np.zeros(0, dtype=np.int64)
            self._index_loaded = True
            return

        # Pre-allocate arrays
        self._embeddings = np.zeros((count, self._embedding_dim), dtype=np.float32)
        self._row_ids = np.zeros(count, dtype=np.int64)

        # Load all embeddings
        cursor = conn.execute("SELECT id, embedding FROM companies WHERE embedding IS NOT NULL")
        for i, row in enumerate(cursor):
            self._row_ids[i] = row["id"]
            self._embeddings[i] = _deserialize_embedding(row["embedding"], self._embedding_dim)

        # Normalize embeddings for cosine similarity (dot product of normalized vectors)
        norms = np.linalg.norm(self._embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        self._embeddings = self._embeddings / norms

        elapsed = time.time() - start
        logger.info(f"Loaded {count} embeddings into memory in {elapsed:.2f}s")

    def close(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def insert(self, record: CompanyRecord, embedding: np.ndarray) -> int:
        """
        Insert a company record with its embedding.

        Args:
            record: Company record to insert
            embedding: Embedding vector for the company name

        Returns:
            Row ID of inserted record
        """
        conn = self._connect()

        # Serialize record and embedding
        record_json = json.dumps(record.record)
        embedding_bytes = _serialize_embedding(embedding)

        cursor = conn.execute("""
            INSERT OR REPLACE INTO companies
            (name, embedding_name, legal_name, source, source_id, region, record, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            record.name,
            record.embedding_name,
            record.legal_name,
            record.source,
            record.source_id,
            record.region,
            record_json,
            embedding_bytes,
        ))

        row_id = cursor.lastrowid
        assert row_id is not None

        # Invalidate in-memory index so it gets reloaded on next search
        self._index_loaded = False

        conn.commit()
        return row_id

    def insert_batch(
        self,
        records: list[CompanyRecord],
        embeddings: np.ndarray,
        batch_size: int = 1000,
    ) -> int:
        """
        Insert multiple company records with embeddings.

        Args:
            records: List of company records
            embeddings: Matrix of embeddings (N x dim)
            batch_size: Commit batch size

        Returns:
            Number of records inserted
        """
        conn = self._connect()
        count = 0

        for record, embedding in zip(records, embeddings):
            record_json = json.dumps(record.record)
            embedding_bytes = _serialize_embedding(embedding)

            conn.execute("""
                INSERT OR REPLACE INTO companies
                (name, embedding_name, legal_name, source, source_id, region, record, embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.name,
                record.embedding_name,
                record.legal_name,
                record.source,
                record.source_id,
                record.region,
                record_json,
                embedding_bytes,
            ))

            count += 1

            if count % batch_size == 0:
                conn.commit()
                logger.info(f"Inserted {count} records...")

        conn.commit()

        # Invalidate in-memory index
        self._index_loaded = False

        return count

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 20,
        source_filter: Optional[str] = None,
    ) -> list[tuple[CompanyRecord, float]]:
        """
        Search for similar companies by embedding.

        Uses in-memory numpy-based search for fast results.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            source_filter: Optional filter by source (gleif, sec_edgar, etc.)

        Returns:
            List of (CompanyRecord, similarity_score) tuples
        """
        start = time.time()

        # Ensure index is loaded
        self._load_index()

        if self._embeddings is None or len(self._embeddings) == 0:
            return []

        # Normalize query embedding
        query_norm = np.linalg.norm(query_embedding)
        if query_norm == 0:
            return []
        query_normalized = query_embedding / query_norm

        # Compute cosine similarities (dot product of normalized vectors)
        similarities = np.dot(self._embeddings, query_normalized)

        # Get top-k indices
        if len(similarities) <= top_k:
            top_indices = np.argsort(similarities)[::-1]
        else:
            # Use argpartition for efficiency with large arrays
            top_indices = np.argpartition(similarities, -top_k)[-top_k:]
            top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]

        # Fetch records from database
        conn = self._connect()
        results = []

        for idx in top_indices:
            row_id = int(self._row_ids[idx])
            similarity = float(similarities[idx])

            cursor = conn.execute("""
                SELECT name, embedding_name, legal_name, source, source_id, region, record
                FROM companies WHERE id = ?
            """, (row_id,))
            row = cursor.fetchone()

            if row is None:
                continue

            # Apply source filter if specified
            if source_filter and row["source"] != source_filter:
                continue

            record = CompanyRecord(
                name=row["name"],
                embedding_name=row["embedding_name"],
                legal_name=row["legal_name"],
                source=row["source"],
                source_id=row["source_id"],
                region=row["region"] or "",
                record=json.loads(row["record"]),
            )
            results.append((record, similarity))

        elapsed = time.time() - start
        logger.debug(f"Vector search took {elapsed:.3f}s (results={len(results)})")
        return results

    def get_by_source_id(self, source: str, source_id: str) -> Optional[CompanyRecord]:
        """Get a company record by source and source_id."""
        conn = self._connect()

        cursor = conn.execute("""
            SELECT name, embedding_name, legal_name, source, source_id, region, record
            FROM companies
            WHERE source = ? AND source_id = ?
        """, (source, source_id))

        row = cursor.fetchone()
        if row:
            return CompanyRecord(
                name=row["name"],
                embedding_name=row["embedding_name"],
                legal_name=row["legal_name"],
                source=row["source"],
                source_id=row["source_id"],
                region=row["region"] or "",
                record=json.loads(row["record"]),
            )
        return None

    def get_stats(self) -> DatabaseStats:
        """Get database statistics."""
        conn = self._connect()

        # Total count
        cursor = conn.execute("SELECT COUNT(*) FROM companies")
        total = cursor.fetchone()[0]

        # Count by source
        cursor = conn.execute("SELECT source, COUNT(*) as cnt FROM companies GROUP BY source")
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
                SELECT name, embedding_name, legal_name, source, source_id, region, record
                FROM companies
                WHERE source = ?
            """, (source,))
        else:
            cursor = conn.execute("""
                SELECT name, embedding_name, legal_name, source, source_id, region, record
                FROM companies
            """)

        for row in cursor:
            yield CompanyRecord(
                name=row["name"],
                embedding_name=row["embedding_name"],
                legal_name=row["legal_name"],
                source=row["source"],
                source_id=row["source_id"],
                region=row["region"] or "",
                record=json.loads(row["record"]),
            )

    def delete_source(self, source: str) -> int:
        """Delete all records from a specific source."""
        conn = self._connect()

        cursor = conn.execute("DELETE FROM companies WHERE source = ?", (source,))
        deleted = cursor.rowcount

        conn.commit()

        # Invalidate in-memory index
        self._index_loaded = False

        logger.info(f"Deleted {deleted} records from source '{source}'")
        return deleted

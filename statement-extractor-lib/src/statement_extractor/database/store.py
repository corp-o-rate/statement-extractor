"""
Company database using sqlite-vec for embedding search.

Provides efficient vector similarity search for company name matching.
"""

import json
import logging
import os
import sqlite3
import struct
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

from .models import CompanyRecord, CompanyMatch, DatabaseStats

logger = logging.getLogger(__name__)

# Default database location
DEFAULT_DB_PATH = Path.home() / ".cache" / "corp-extractor" / "companies.db"


def _serialize_embedding(embedding: np.ndarray) -> bytes:
    """Serialize numpy array to bytes for sqlite-vec."""
    return struct.pack(f"{len(embedding)}f", *embedding.tolist())


def _deserialize_embedding(data: bytes, dim: int) -> np.ndarray:
    """Deserialize bytes to numpy array."""
    return np.array(struct.unpack(f"{dim}f", data), dtype=np.float32)


class CompanyDatabase:
    """
    SQLite database with vector search for companies.

    Uses sqlite-vec extension for efficient embedding similarity search.
    Falls back to brute-force search if sqlite-vec is not available.
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
        self._has_vec_extension = False

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

        # Try to load sqlite-vec extension
        self._has_vec_extension = self._load_vec_extension()

        # Create tables
        self._create_tables()

        return self._conn

    def _load_vec_extension(self) -> bool:
        """Try to load sqlite-vec extension."""
        try:
            import sqlite_vec

            self._conn.enable_load_extension(True)
            sqlite_vec.load(self._conn)
            self._conn.enable_load_extension(False)
            logger.debug("sqlite-vec extension loaded successfully")
            return True
        except ImportError:
            logger.debug("sqlite-vec not installed, using fallback search")
            return False
        except Exception as e:
            logger.debug(f"Failed to load sqlite-vec: {e}")
            return False

    def _create_tables(self) -> None:
        """Create database tables."""
        conn = self._conn

        # Main company records table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS companies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                embedding_name TEXT NOT NULL,
                legal_name TEXT NOT NULL,
                source TEXT NOT NULL,
                source_id TEXT NOT NULL,
                record TEXT NOT NULL,
                embedding BLOB,
                UNIQUE(source, source_id)
            )
        """)

        # Create indexes
        conn.execute("CREATE INDEX IF NOT EXISTS idx_companies_name ON companies(name)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_companies_source ON companies(source)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_companies_source_id ON companies(source, source_id)")

        # Create virtual table for vector search if extension available
        if self._has_vec_extension:
            try:
                conn.execute(f"""
                    CREATE VIRTUAL TABLE IF NOT EXISTS company_embeddings USING vec0(
                        company_id INTEGER PRIMARY KEY,
                        embedding float[{self._embedding_dim}]
                    )
                """)
                logger.debug("Created vec0 virtual table for embeddings")
            except sqlite3.OperationalError as e:
                logger.warning(f"Failed to create vec0 table: {e}")
                self._has_vec_extension = False

        conn.commit()

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
            (name, embedding_name, legal_name, source, source_id, record, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            record.name,
            record.embedding_name,
            record.legal_name,
            record.source,
            record.source_id,
            record_json,
            embedding_bytes,
        ))

        row_id = cursor.lastrowid

        # Insert into vector table if available
        if self._has_vec_extension:
            try:
                conn.execute("""
                    INSERT OR REPLACE INTO company_embeddings (company_id, embedding)
                    VALUES (?, ?)
                """, (row_id, embedding_bytes))
            except sqlite3.OperationalError as e:
                logger.debug(f"Failed to insert into vec table: {e}")

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

        for i, (record, embedding) in enumerate(zip(records, embeddings)):
            record_json = json.dumps(record.record)
            embedding_bytes = _serialize_embedding(embedding)

            cursor = conn.execute("""
                INSERT OR REPLACE INTO companies
                (name, embedding_name, legal_name, source, source_id, record, embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                record.name,
                record.embedding_name,
                record.legal_name,
                record.source,
                record.source_id,
                record_json,
                embedding_bytes,
            ))

            row_id = cursor.lastrowid

            if self._has_vec_extension:
                try:
                    conn.execute("""
                        INSERT OR REPLACE INTO company_embeddings (company_id, embedding)
                        VALUES (?, ?)
                    """, (row_id, embedding_bytes))
                except sqlite3.OperationalError:
                    pass

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
    ) -> list[tuple[CompanyRecord, float]]:
        """
        Search for similar companies by embedding.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            source_filter: Optional filter by source (gleif, sec_edgar, etc.)

        Returns:
            List of (CompanyRecord, similarity_score) tuples
        """
        conn = self._connect()

        if self._has_vec_extension:
            return self._search_with_vec(query_embedding, top_k, source_filter)
        else:
            return self._search_brute_force(query_embedding, top_k, source_filter)

    def _search_with_vec(
        self,
        query_embedding: np.ndarray,
        top_k: int,
        source_filter: Optional[str],
    ) -> list[tuple[CompanyRecord, float]]:
        """Search using sqlite-vec extension."""
        conn = self._conn
        query_bytes = _serialize_embedding(query_embedding)

        # Use vec_distance_cosine for similarity search
        if source_filter:
            cursor = conn.execute("""
                SELECT c.*, vec_distance_cosine(ce.embedding, ?) as distance
                FROM company_embeddings ce
                JOIN companies c ON c.id = ce.company_id
                WHERE c.source = ?
                ORDER BY distance ASC
                LIMIT ?
            """, (query_bytes, source_filter, top_k))
        else:
            cursor = conn.execute("""
                SELECT c.*, vec_distance_cosine(ce.embedding, ?) as distance
                FROM company_embeddings ce
                JOIN companies c ON c.id = ce.company_id
                ORDER BY distance ASC
                LIMIT ?
            """, (query_bytes, top_k))

        results = []
        for row in cursor:
            record = CompanyRecord(
                name=row["name"],
                embedding_name=row["embedding_name"],
                legal_name=row["legal_name"],
                source=row["source"],
                source_id=row["source_id"],
                record=json.loads(row["record"]),
            )
            # Convert distance to similarity (1 - distance for cosine)
            similarity = 1.0 - row["distance"]
            results.append((record, similarity))

        return results

    def _search_brute_force(
        self,
        query_embedding: np.ndarray,
        top_k: int,
        source_filter: Optional[str],
    ) -> list[tuple[CompanyRecord, float]]:
        """Fallback brute-force search without vec extension."""
        conn = self._conn

        # Load all embeddings
        if source_filter:
            cursor = conn.execute("""
                SELECT id, name, embedding_name, legal_name, source, source_id, record, embedding
                FROM companies
                WHERE source = ?
            """, (source_filter,))
        else:
            cursor = conn.execute("""
                SELECT id, name, embedding_name, legal_name, source, source_id, record, embedding
                FROM companies
            """)

        records = []
        embeddings = []
        for row in cursor:
            records.append(CompanyRecord(
                name=row["name"],
                embedding_name=row["embedding_name"],
                legal_name=row["legal_name"],
                source=row["source"],
                source_id=row["source_id"],
                record=json.loads(row["record"]),
            ))
            embeddings.append(_deserialize_embedding(row["embedding"], self._embedding_dim))

        if not embeddings:
            return []

        # Compute similarities
        embeddings_matrix = np.array(embeddings)
        similarities = np.dot(embeddings_matrix, query_embedding)

        # Get top-k indices
        if len(similarities) <= top_k:
            indices = np.argsort(similarities)[::-1]
        else:
            indices = np.argpartition(similarities, -top_k)[-top_k:]
            indices = indices[np.argsort(similarities[indices])[::-1]]

        return [(records[i], float(similarities[i])) for i in indices]

    def get_by_source_id(self, source: str, source_id: str) -> Optional[CompanyRecord]:
        """Get a company record by source and source_id."""
        conn = self._connect()

        cursor = conn.execute("""
            SELECT name, embedding_name, legal_name, source, source_id, record
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
                SELECT name, embedding_name, legal_name, source, source_id, record
                FROM companies
                WHERE source = ?
            """, (source,))
        else:
            cursor = conn.execute("""
                SELECT name, embedding_name, legal_name, source, source_id, record
                FROM companies
            """)

        for row in cursor:
            yield CompanyRecord(
                name=row["name"],
                embedding_name=row["embedding_name"],
                legal_name=row["legal_name"],
                source=row["source"],
                source_id=row["source_id"],
                record=json.loads(row["record"]),
            )

    def delete_source(self, source: str) -> int:
        """Delete all records from a specific source."""
        conn = self._connect()

        cursor = conn.execute("DELETE FROM companies WHERE source = ?", (source,))
        deleted = cursor.rowcount

        if self._has_vec_extension:
            # Also delete from vector table
            conn.execute("""
                DELETE FROM company_embeddings
                WHERE company_id NOT IN (SELECT id FROM companies)
            """)

        conn.commit()
        logger.info(f"Deleted {deleted} records from source '{source}'")
        return deleted

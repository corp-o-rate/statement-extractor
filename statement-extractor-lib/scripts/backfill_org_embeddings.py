#!/usr/bin/env python3
"""
Backfill embeddings for organizations that don't have them.

This script finds all organizations without embeddings and generates them
using the CompanyEmbedder.
"""

import logging
import sqlite3
import sys
from pathlib import Path

import numpy as np
import sqlite_vec

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def backfill_embeddings(db_path: Path, batch_size: int = 1000, limit: int | None = None) -> dict:
    """Backfill embeddings for organizations without them."""
    from statement_extractor.database.embeddings import CompanyEmbedder

    # Connect with sqlite-vec
    conn = sqlite3.connect(str(db_path))
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.row_factory = sqlite3.Row

    stats = {
        "total_without_embeddings": 0,
        "processed": 0,
        "embedded": 0,
        "errors": 0,
    }

    # Count organizations without embeddings
    cursor = conn.execute("""
        SELECT COUNT(*) FROM organizations o
        WHERE NOT EXISTS (SELECT 1 FROM organization_embeddings e WHERE e.org_id = o.id)
    """)
    stats["total_without_embeddings"] = cursor.fetchone()[0]
    logger.info(f"Found {stats['total_without_embeddings']:,} organizations without embeddings")

    if stats["total_without_embeddings"] == 0:
        logger.info("No organizations need embeddings!")
        return stats

    # Load embedder
    logger.info("Loading embedding model...")
    embedder = CompanyEmbedder()
    logger.info("Embedding model loaded.")

    # Process in batches
    offset = 0
    target = limit if limit else stats["total_without_embeddings"]

    while stats["embedded"] < target:
        # Get batch of orgs without embeddings
        cursor = conn.execute("""
            SELECT o.id, o.name FROM organizations o
            WHERE NOT EXISTS (SELECT 1 FROM organization_embeddings e WHERE e.org_id = o.id)
            LIMIT ?
        """, (batch_size,))

        rows = cursor.fetchall()
        if not rows:
            break

        stats["processed"] += len(rows)

        # Extract names and IDs
        ids = [row["id"] for row in rows]
        names = [row["name"] for row in rows]

        # Generate embeddings
        try:
            embeddings = embedder.embed_batch(names)

            # Insert embeddings
            for org_id, embedding in zip(ids, embeddings):
                embedding_blob = embedding.astype(np.float32).tobytes()
                conn.execute("""
                    INSERT OR REPLACE INTO organization_embeddings (org_id, embedding)
                    VALUES (?, ?)
                """, (org_id, embedding_blob))

            conn.commit()
            stats["embedded"] += len(rows)
            logger.info(f"Progress: {stats['embedded']:,} / {target:,} ({100 * stats['embedded'] / target:.1f}%)")

        except Exception as e:
            logger.error(f"Error embedding batch: {e}")
            stats["errors"] += 1
            conn.rollback()

        if limit and stats["embedded"] >= limit:
            break

    conn.close()
    return stats


def main():
    db_path = Path.home() / ".cache" / "corp-extractor" / "entities-v2.db"
    batch_size = 1000
    limit = None

    # Parse args
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--db" and i + 1 < len(args):
            db_path = Path(args[i + 1])
            i += 2
        elif args[i] == "--batch-size" and i + 1 < len(args):
            batch_size = int(args[i + 1])
            i += 2
        elif args[i] == "--limit" and i + 1 < len(args):
            limit = int(args[i + 1])
            i += 2
        else:
            i += 1

    if not db_path.exists():
        logger.error(f"Database not found: {db_path}")
        return 1

    logger.info(f"Backfilling embeddings in {db_path}")
    logger.info(f"Batch size: {batch_size}")
    if limit:
        logger.info(f"Limit: {limit:,}")

    try:
        stats = backfill_embeddings(db_path, batch_size=batch_size, limit=limit)
        logger.info(f"Done! Stats: {stats}")
        return 0
    except Exception as e:
        logger.error(f"Failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

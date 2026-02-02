#!/usr/bin/env python3
"""
Resolve parent_ids in the locations table.

This script:
1. Reads all locations with parent_qids in their record field
2. Looks up the parent QIDs in the database to get their location IDs
3. Updates the parent_ids field with the resolved IDs
"""

import json
import logging
import sqlite3
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def resolve_parent_ids(db_path: Path) -> dict:
    """Resolve parent_ids from parent_qids in location records."""

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    stats = {
        "total_locations": 0,
        "with_parent_qids": 0,
        "resolved_parents": 0,
        "partially_resolved": 0,
        "updated_records": 0,
    }

    # Build QID -> location_id lookup
    logger.info("Building QID to location_id lookup...")
    cursor = conn.execute("SELECT id, qid FROM locations WHERE qid IS NOT NULL")
    qid_to_id: dict[int, int] = {}
    for row in cursor:
        qid_to_id[row["qid"]] = row["id"]
    logger.info(f"Built lookup with {len(qid_to_id):,} QID -> location_id mappings")

    # Get all locations with parent_qids in their record
    logger.info("Reading locations with parent_qids...")
    cursor = conn.execute("""
        SELECT id, qid, name, record, parent_ids
        FROM locations
        WHERE record LIKE '%parent_qids%'
    """)

    updates = []
    for row in cursor:
        stats["total_locations"] += 1

        try:
            record = json.loads(row["record"]) if row["record"] else {}
        except json.JSONDecodeError:
            continue

        parent_qids = record.get("parent_qids", [])
        country_qids = record.get("country_qids", [])

        if not parent_qids and not country_qids:
            continue

        stats["with_parent_qids"] += 1

        # Resolve parent QIDs to IDs
        resolved_ids = []
        unresolved_count = 0

        # First add parent_qids (more specific hierarchy)
        for qid_str in parent_qids:
            if isinstance(qid_str, str) and qid_str.startswith("Q"):
                try:
                    qid_int = int(qid_str[1:])
                    if qid_int in qid_to_id:
                        resolved_ids.append(qid_to_id[qid_int])
                    else:
                        unresolved_count += 1
                except ValueError:
                    pass

        # Then add country_qids as fallback (if not already present via parent chain)
        for qid_str in country_qids:
            if isinstance(qid_str, str) and qid_str.startswith("Q"):
                try:
                    qid_int = int(qid_str[1:])
                    if qid_int in qid_to_id:
                        parent_id = qid_to_id[qid_int]
                        if parent_id not in resolved_ids:
                            resolved_ids.append(parent_id)
                    else:
                        unresolved_count += 1
                except ValueError:
                    pass

        if resolved_ids:
            # Deduplicate while preserving order
            seen = set()
            unique_ids = []
            for pid in resolved_ids:
                if pid not in seen:
                    seen.add(pid)
                    unique_ids.append(pid)

            updates.append((json.dumps(unique_ids), row["id"]))

            if unresolved_count > 0:
                stats["partially_resolved"] += 1
            else:
                stats["resolved_parents"] += 1

    logger.info(f"Found {len(updates):,} locations to update with parent_ids")

    # Update in batches
    if updates:
        logger.info("Updating parent_ids...")
        batch_size = 10000
        for i in range(0, len(updates), batch_size):
            batch = updates[i:i + batch_size]
            conn.executemany(
                "UPDATE locations SET parent_ids = ? WHERE id = ?",
                batch
            )
            conn.commit()
            stats["updated_records"] += len(batch)
            logger.info(f"  Updated {stats['updated_records']:,} records...")

    conn.close()
    return stats


def main():
    db_path = Path.home() / ".cache" / "corp-extractor" / "entities-v2.db"

    if len(sys.argv) > 1:
        db_path = Path(sys.argv[1])

    if not db_path.exists():
        logger.error(f"Database not found: {db_path}")
        return 1

    logger.info(f"Resolving parent_ids in {db_path}")

    try:
        stats = resolve_parent_ids(db_path)
        logger.info(f"Done! Stats: {stats}")
        return 0
    except Exception as e:
        logger.error(f"Failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

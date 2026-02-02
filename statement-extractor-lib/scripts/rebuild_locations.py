#!/usr/bin/env python3
"""
Rebuild the locations table with all unique regions/countries from the v1 database.

This script:
1. Extracts all unique regions from v1 organizations + countries from v1 people
2. Creates a location entry for EACH unique string (not normalized)
3. Looks up QIDs from qid_labels
4. Updates v2 organization region_ids using v1 source_identifier mapping
5. Updates v2 people country_ids using v1 source_identifier mapping
"""

import json
import logging
import sqlite3
import sys
from pathlib import Path
from typing import Optional

import pycountry

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Source IDs
SOURCE_PYCOUNTRY = 5
SOURCE_WIKIDATA = 4

# Location type IDs (from seed_data)
LOCATION_TYPE_COUNTRY = 2


def normalize_for_lookup(text: str) -> str:
    """Normalize text for lookup matching."""
    return text.lower().strip()


def parse_qid_to_int(qid_str: str) -> Optional[int]:
    """Parse a QID string like 'Q30' to integer 30."""
    if qid_str and qid_str.startswith("Q") and qid_str[1:].isdigit():
        return int(qid_str[1:])
    return None


def get_pycountry_info(text: str) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Try to resolve text to pycountry country info.
    Returns (alpha_2, alpha_3, official_name) or (None, None, None)
    """
    text_clean = text.strip()
    if not text_clean:
        return None, None, None

    # Try alpha_2
    if len(text_clean) == 2:
        try:
            country = pycountry.countries.get(alpha_2=text_clean.upper())
            if country:
                return country.alpha_2, getattr(country, 'alpha_3', None), country.name
        except Exception:
            pass

    # Try alpha_3
    if len(text_clean) == 3:
        try:
            country = pycountry.countries.get(alpha_3=text_clean.upper())
            if country:
                return country.alpha_2, country.alpha_3, country.name
        except Exception:
            pass

    # Try fuzzy search by name
    try:
        matches = pycountry.countries.search_fuzzy(text_clean)
        if matches:
            country = matches[0]
            return country.alpha_2, getattr(country, 'alpha_3', None), country.name
    except Exception:
        pass

    return None, None, None


def rebuild_locations(v1_path: Path, v2_path: Path) -> dict:
    """Rebuild the locations table from v1 data."""

    v1_conn = sqlite3.connect(str(v1_path))
    v1_conn.row_factory = sqlite3.Row

    v2_conn = sqlite3.connect(str(v2_path))
    v2_conn.row_factory = sqlite3.Row

    stats = {
        "unique_regions": 0,
        "unique_countries": 0,
        "total_locations": 0,
        "with_qid": 0,
        "with_pycountry": 0,
        "orgs_updated": 0,
        "people_updated": 0,
    }

    # Build QID label index (lowercase label -> QID string)
    logger.info("Building QID label index from v1 qid_labels...")
    cursor = v1_conn.execute("SELECT qid, label FROM qid_labels")
    label_to_qid: dict[str, str] = {}
    count = 0
    for row in cursor:
        qid, label = row[0], row[1]
        if label:
            label_lower = label.lower().strip()
            if label_lower not in label_to_qid:
                label_to_qid[label_lower] = qid
        count += 1
        if count % 5000000 == 0:
            logger.info(f"  Processed {count} labels...")
    logger.info(f"Built index with {len(label_to_qid)} unique labels from {count} total")

    # Collect unique regions from organizations
    logger.info("Collecting unique regions from v1 organizations...")
    cursor = v1_conn.execute("""
        SELECT DISTINCT region FROM organizations
        WHERE region IS NOT NULL AND length(trim(region)) > 0
    """)
    regions = {row[0].strip() for row in cursor if row[0]}
    stats["unique_regions"] = len(regions)
    logger.info(f"Found {len(regions)} unique regions")

    # Collect unique countries from people
    logger.info("Collecting unique countries from v1 people...")
    cursor = v1_conn.execute("""
        SELECT DISTINCT country FROM people
        WHERE country IS NOT NULL AND length(trim(country)) > 0
    """)
    countries = {row[0].strip() for row in cursor if row[0]}
    stats["unique_countries"] = len(countries)
    logger.info(f"Found {len(countries)} unique countries")

    # Combine all unique location strings
    all_locations = regions | countries
    logger.info(f"Total unique location strings: {len(all_locations)}")

    # Clear existing locations table
    logger.info("Clearing existing locations table...")
    v2_conn.execute("DELETE FROM locations")
    v2_conn.commit()

    # Insert all locations and build name -> id mapping
    logger.info("Inserting locations...")
    name_to_id: dict[str, int] = {}

    for i, loc_name in enumerate(sorted(all_locations)):
        name_normalized = normalize_for_lookup(loc_name)

        # Look up QID
        qid = None
        qid_str = label_to_qid.get(name_normalized)
        if qid_str:
            qid = parse_qid_to_int(qid_str)
            if qid:
                stats["with_qid"] += 1

        # Get pycountry info
        alpha_2, alpha_3, official_name = get_pycountry_info(loc_name)

        if alpha_2:
            stats["with_pycountry"] += 1

        # Always use the original location name as source_identifier
        # This ensures each unique region/country string gets its own entry
        source_id = SOURCE_WIKIDATA if qid else SOURCE_PYCOUNTRY
        source_identifier = loc_name

        # Build record with extra info
        record: dict = {}
        if alpha_2:
            record["alpha_2"] = alpha_2
        if alpha_3:
            record["alpha_3"] = alpha_3
        if official_name and official_name != loc_name:
            record["official_name"] = official_name

        # Insert location with JSONB parent_ids
        cursor = v2_conn.execute("""
            INSERT INTO locations
            (qid, name, name_normalized, source_id, source_identifier, location_type_id, parent_ids, record)
            VALUES (?, ?, ?, ?, ?, ?, json(?), ?)
        """, (
            qid,
            loc_name,
            name_normalized,
            source_id,
            source_identifier,
            LOCATION_TYPE_COUNTRY,
            "[]",  # Empty JSON array for parent_ids
            json.dumps(record),
        ))

        location_id = cursor.lastrowid
        assert location_id is not None
        name_to_id[loc_name] = location_id

        stats["total_locations"] += 1

        if (i + 1) % 1000 == 0:
            v2_conn.commit()
            logger.info(f"  Inserted {i + 1} locations...")

    v2_conn.commit()
    logger.info(f"Inserted {stats['total_locations']} locations")

    # Now update organization region_ids by matching source_identifier
    logger.info("Updating organization region_ids...")
    logger.info("This requires matching v1 source+source_id to v2 source_id+source_identifier")

    # Process organizations in batches
    batch_size = 50000
    offset = 0

    while True:
        # Get batch of v1 organizations with their regions
        cursor = v1_conn.execute("""
            SELECT source, source_id, region
            FROM organizations
            WHERE region IS NOT NULL AND length(trim(region)) > 0
            LIMIT ? OFFSET ?
        """, (batch_size, offset))

        rows = cursor.fetchall()
        if not rows:
            break

        # Build updates
        for row in rows:
            source, source_id, region = row["source"], row["source_id"], row["region"]
            if not region:
                continue

            region = region.strip()
            location_id = name_to_id.get(region)
            if not location_id:
                continue

            # Map v1 source name to v2 source_id
            source_type_id = {"gleif": 1, "sec_edgar": 2, "companies_house": 3, "wikipedia": 4, "wikidata": 4}.get(source, 4)

            # Update v2 organization
            v2_conn.execute("""
                UPDATE organizations
                SET region_id = ?
                WHERE source_id = ? AND source_identifier = ?
            """, (location_id, source_type_id, source_id))

            stats["orgs_updated"] += 1

        v2_conn.commit()
        offset += batch_size
        logger.info(f"  Updated {stats['orgs_updated']} organizations...")

    # Update people country_ids
    logger.info("Updating people country_ids...")
    offset = 0

    while True:
        cursor = v1_conn.execute("""
            SELECT source, source_id, country
            FROM people
            WHERE country IS NOT NULL AND length(trim(country)) > 0
            LIMIT ? OFFSET ?
        """, (batch_size, offset))

        rows = cursor.fetchall()
        if not rows:
            break

        for row in rows:
            source, source_id, country = row["source"], row["source_id"], row["country"]
            if not country:
                continue

            country = country.strip()
            location_id = name_to_id.get(country)
            if not location_id:
                continue

            source_type_id = {"gleif": 1, "sec_edgar": 2, "companies_house": 3, "wikipedia": 4, "wikidata": 4}.get(source, 4)

            v2_conn.execute("""
                UPDATE people
                SET country_id = ?
                WHERE source_id = ? AND source_identifier = ?
            """, (location_id, source_type_id, source_id))

            stats["people_updated"] += 1

        v2_conn.commit()
        offset += batch_size
        logger.info(f"  Updated {stats['people_updated']} people...")

    v1_conn.close()
    v2_conn.close()

    return stats


def main():
    v1_path = Path.home() / ".cache" / "corp-extractor" / "entities.db"
    v2_path = Path.home() / ".cache" / "corp-extractor" / "entities-v2.db"

    if len(sys.argv) > 2:
        v1_path = Path(sys.argv[1])
        v2_path = Path(sys.argv[2])

    if not v1_path.exists():
        logger.error(f"V1 database not found: {v1_path}")
        return 1

    if not v2_path.exists():
        logger.error(f"V2 database not found: {v2_path}")
        return 1

    logger.info(f"Rebuilding locations from {v1_path} into {v2_path}")

    try:
        stats = rebuild_locations(v1_path, v2_path)
        logger.info(f"Done! Stats: {stats}")
        return 0
    except Exception as e:
        logger.error(f"Failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

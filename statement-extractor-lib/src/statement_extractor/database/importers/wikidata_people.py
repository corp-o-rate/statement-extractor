"""
Wikidata importer for the person database.

Imports notable people data from Wikidata using SPARQL queries
into the embedding database for person name matching.

Uses a two-phase approach for reliability:
1. Bulk fetch: Simple queries to get QID + name + country (fast, no timeouts)
2. Enrich: Targeted per-person queries for role/org/dates (resumable)

Notable people are those with English Wikipedia articles, ensuring
a basic level of notability.

Query categories (organized by PersonType):
- executives: Business executives (CEOs, CFOs, etc.)
- politicians: Politicians and diplomats
- athletes: Sports figures
- artists: Actors, musicians, directors
- academics: Professors and researchers
- scientists: Scientists and inventors
- journalists: Media personalities
- entrepreneurs: Founders and business owners

Uses the public Wikidata Query Service endpoint.
"""

import json
import logging
import time
import urllib.parse
import urllib.request
from typing import Any, Iterator, Optional

from ..models import CompanyRecord, EntityType, PersonRecord, PersonType

logger = logging.getLogger(__name__)

# Wikidata SPARQL endpoint
WIKIDATA_SPARQL_URL = "https://query.wikidata.org/sparql"

# =============================================================================
# BULK QUERIES - Simple, fast queries for initial import (no role/org/dates)
# =============================================================================

# Bulk query for executives - just get people who held executive positions
BULK_EXECUTIVE_QUERY = """
SELECT DISTINCT ?person ?personLabel ?countryLabel ?description WHERE {
  ?person wdt:P31 wd:Q5 .
  ?person wdt:P39 ?role .
  VALUES ?role {
    wd:Q484876 wd:Q623279 wd:Q1502675 wd:Q935019 wd:Q1057716 wd:Q2140589
    wd:Q1115042 wd:Q4720025 wd:Q60432825 wd:Q15967139 wd:Q15729310 wd:Q47523568
    wd:Q258557 wd:Q114863313 wd:Q726114 wd:Q1372944 wd:Q18918145 wd:Q1057569
    wd:Q24058752 wd:Q3578048 wd:Q476675 wd:Q5441744 wd:Q4188234 wd:Q38844673
    wd:Q97273203 wd:Q60715311 wd:Q3563879 wd:Q3505845
  }
  OPTIONAL { ?person wdt:P27 ?country . }
  OPTIONAL { ?person schema:description ?description FILTER(LANG(?description) = "en") }
  ?article schema:about ?person ; schema:isPartOf <https://en.wikipedia.org/> .
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
LIMIT %d
OFFSET %d
"""

# Bulk query for politicians
BULK_POLITICIAN_QUERY = """
SELECT DISTINCT ?person ?personLabel ?countryLabel ?description WHERE {
  ?person wdt:P31 wd:Q5 .
  ?person wdt:P39 ?role .
  VALUES ?roleType { wd:Q2285706 wd:Q30461 wd:Q83307 wd:Q4175034 }
  ?role wdt:P31 ?roleType .
  OPTIONAL { ?person wdt:P27 ?country . }
  OPTIONAL { ?person schema:description ?description FILTER(LANG(?description) = "en") }
  ?article schema:about ?person ; schema:isPartOf <https://en.wikipedia.org/> .
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
LIMIT %d
OFFSET %d
"""

# Bulk query for athletes
BULK_ATHLETE_QUERY = """
SELECT DISTINCT ?person ?personLabel ?countryLabel ?description WHERE {
  ?person wdt:P31 wd:Q5 .
  ?person wdt:P106 wd:Q2066131 .
  OPTIONAL { ?person wdt:P27 ?country . }
  OPTIONAL { ?person schema:description ?description FILTER(LANG(?description) = "en") }
  ?article schema:about ?person ; schema:isPartOf <https://en.wikipedia.org/> .
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
LIMIT %d
OFFSET %d
"""

# Bulk query for artists
BULK_ARTIST_QUERY = """
SELECT DISTINCT ?person ?personLabel ?countryLabel ?description WHERE {
  ?person wdt:P31 wd:Q5 .
  ?person wdt:P106 ?occupation .
  VALUES ?occupation { wd:Q33999 wd:Q177220 wd:Q639669 wd:Q2526255 wd:Q36180 wd:Q483501 }
  OPTIONAL { ?person wdt:P27 ?country . }
  OPTIONAL { ?person schema:description ?description FILTER(LANG(?description) = "en") }
  ?article schema:about ?person ; schema:isPartOf <https://en.wikipedia.org/> .
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
LIMIT %d
OFFSET %d
"""

# Bulk query for academics
BULK_ACADEMIC_QUERY = """
SELECT DISTINCT ?person ?personLabel ?countryLabel ?description WHERE {
  ?person wdt:P31 wd:Q5 .
  { ?person wdt:P106 wd:Q121594 . } UNION { ?person wdt:P106 wd:Q3400985 . }
  OPTIONAL { ?person wdt:P27 ?country . }
  OPTIONAL { ?person schema:description ?description FILTER(LANG(?description) = "en") }
  ?article schema:about ?person ; schema:isPartOf <https://en.wikipedia.org/> .
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
LIMIT %d
OFFSET %d
"""

# Bulk query for scientists
BULK_SCIENTIST_QUERY = """
SELECT DISTINCT ?person ?personLabel ?countryLabel ?description WHERE {
  ?person wdt:P31 wd:Q5 .
  { ?person wdt:P106 wd:Q901 . } UNION { ?person wdt:P106 wd:Q1650915 . }
  OPTIONAL { ?person wdt:P27 ?country . }
  OPTIONAL { ?person schema:description ?description FILTER(LANG(?description) = "en") }
  ?article schema:about ?person ; schema:isPartOf <https://en.wikipedia.org/> .
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
LIMIT %d
OFFSET %d
"""

# Bulk query for journalists
BULK_JOURNALIST_QUERY = """
SELECT DISTINCT ?person ?personLabel ?countryLabel ?description WHERE {
  ?person wdt:P31 wd:Q5 .
  { ?person wdt:P106 wd:Q1930187 . } UNION { ?person wdt:P106 wd:Q13590141 . }
  OPTIONAL { ?person wdt:P27 ?country . }
  OPTIONAL { ?person schema:description ?description FILTER(LANG(?description) = "en") }
  ?article schema:about ?person ; schema:isPartOf <https://en.wikipedia.org/> .
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
LIMIT %d
OFFSET %d
"""

# Bulk query for entrepreneurs
BULK_ENTREPRENEUR_QUERY = """
SELECT DISTINCT ?person ?personLabel ?countryLabel ?description WHERE {
  ?person wdt:P31 wd:Q5 .
  ?org wdt:P112 ?person .
  OPTIONAL { ?person wdt:P27 ?country . }
  OPTIONAL { ?person schema:description ?description FILTER(LANG(?description) = "en") }
  ?article schema:about ?person ; schema:isPartOf <https://en.wikipedia.org/> .
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
LIMIT %d
OFFSET %d
"""

# Bulk query for activists
BULK_ACTIVIST_QUERY = """
SELECT DISTINCT ?person ?personLabel ?countryLabel ?description WHERE {
  ?person wdt:P31 wd:Q5 .
  ?person wdt:P106 wd:Q15253558 .
  OPTIONAL { ?person wdt:P27 ?country . }
  OPTIONAL { ?person schema:description ?description FILTER(LANG(?description) = "en") }
  ?article schema:about ?person ; schema:isPartOf <https://en.wikipedia.org/> .
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
LIMIT %d
OFFSET %d
"""

# Mapping query type to PersonType
QUERY_TYPE_TO_PERSON_TYPE: dict[str, PersonType] = {
    "executive": PersonType.EXECUTIVE,
    "politician": PersonType.POLITICIAN,
    "athlete": PersonType.ATHLETE,
    "artist": PersonType.ARTIST,
    "academic": PersonType.ACADEMIC,
    "scientist": PersonType.SCIENTIST,
    "journalist": PersonType.JOURNALIST,
    "entrepreneur": PersonType.ENTREPRENEUR,
    "activist": PersonType.ACTIVIST,
}

# Mapping query type to bulk SPARQL query template
BULK_QUERY_TYPES: dict[str, str] = {
    "executive": BULK_EXECUTIVE_QUERY,
    "politician": BULK_POLITICIAN_QUERY,
    "athlete": BULK_ATHLETE_QUERY,
    "artist": BULK_ARTIST_QUERY,
    "academic": BULK_ACADEMIC_QUERY,
    "scientist": BULK_SCIENTIST_QUERY,
    "journalist": BULK_JOURNALIST_QUERY,
    "entrepreneur": BULK_ENTREPRENEUR_QUERY,
    "activist": BULK_ACTIVIST_QUERY,
}


class WikidataPeopleImporter:
    """
    Importer for Wikidata person data.

    Uses SPARQL queries against the public Wikidata Query Service
    to fetch notable people including executives, politicians, athletes, etc.

    Query types:
    - executive: Business executives (CEOs, CFOs, etc.)
    - politician: Politicians and diplomats
    - athlete: Sports figures
    - artist: Actors, musicians, directors, writers
    - academic: Professors and researchers
    - scientist: Scientists and inventors
    - journalist: Media personalities
    - entrepreneur: Company founders
    - activist: Activists and advocates
    """

    def __init__(
        self,
        batch_size: int = 500,
        delay_seconds: float = 2.0,
        timeout: int = 120,
        max_retries: int = 3,
        min_batch_size: int = 50,
    ):
        """
        Initialize the Wikidata people importer.

        Args:
            batch_size: Number of records to fetch per SPARQL query (default 500)
            delay_seconds: Delay between requests to be polite to the endpoint
            timeout: HTTP timeout in seconds (default 120)
            max_retries: Maximum retries per batch on timeout (default 3)
            min_batch_size: Minimum batch size before giving up (default 50)
        """
        self._batch_size = batch_size
        self._delay = delay_seconds
        self._timeout = timeout
        self._max_retries = max_retries
        self._min_batch_size = min_batch_size
        # Track discovered organizations: org_qid -> org_label
        self._discovered_orgs: dict[str, str] = {}

    def import_from_sparql(
        self,
        limit: Optional[int] = None,
        query_type: str = "executive",
        import_all: bool = False,
    ) -> Iterator[PersonRecord]:
        """
        Import person records from Wikidata via SPARQL (bulk fetch phase).

        This performs the fast bulk import with minimal data (QID, name, country).
        Use enrich_people_batch() afterwards to add role/org/dates.

        Args:
            limit: Optional limit on total records
            query_type: Which query to use (executive, politician, athlete, etc.)
            import_all: If True, run all query types sequentially

        Yields:
            PersonRecord for each person (without role/org - use enrich to add)
        """
        if import_all:
            yield from self._import_all_types(limit)
            return

        if query_type not in BULK_QUERY_TYPES:
            raise ValueError(f"Unknown query type: {query_type}. Use one of: {list(BULK_QUERY_TYPES.keys())}")

        query_template = BULK_QUERY_TYPES[query_type]
        person_type = QUERY_TYPE_TO_PERSON_TYPE.get(query_type, PersonType.UNKNOWN)
        logger.info(f"Starting Wikidata bulk import (query_type={query_type}, person_type={person_type.value})...")

        offset = 0
        total_count = 0
        # Track seen QIDs to deduplicate
        seen_qids: set[str] = set()
        # Current batch size (may be reduced on timeouts)
        current_batch_size = self._batch_size

        while True:
            if limit and total_count >= limit:
                break

            batch_limit = min(current_batch_size, (limit - total_count) if limit else current_batch_size)

            logger.info(f"Fetching Wikidata batch at offset {offset} (batch_size={batch_limit})...")

            # Retry with smaller batch sizes on timeout
            results = None
            retries = 0
            retry_batch_size = batch_limit

            while retries <= self._max_retries:
                try:
                    query = query_template % (retry_batch_size, offset)
                    results = self._execute_sparql(query)
                    # Success - keep the reduced batch size for future requests
                    if retry_batch_size < current_batch_size:
                        current_batch_size = retry_batch_size
                        logger.info(f"Batch succeeded, continuing with batch_size={current_batch_size}")
                    break
                except Exception as e:
                    is_timeout = "timeout" in str(e).lower() or "504" in str(e) or "503" in str(e)
                    if is_timeout and retry_batch_size > self._min_batch_size:
                        retries += 1
                        retry_batch_size = max(retry_batch_size // 2, self._min_batch_size)
                        wait_time = self._delay * (2 ** retries)  # Exponential backoff
                        logger.warning(
                            f"Timeout at offset {offset}, retry {retries}/{self._max_retries} "
                            f"with batch_size={retry_batch_size} after {wait_time:.1f}s wait"
                        )
                        time.sleep(wait_time)
                    else:
                        logger.error(f"SPARQL query failed at offset {offset}: {e}")
                        break

            if results is None:
                logger.error(f"Giving up on offset {offset} after {retries} retries")
                break

            bindings = results.get("results", {}).get("bindings", [])

            if not bindings:
                logger.info("No more results from Wikidata")
                break

            batch_count = 0
            skipped_count = 0
            for binding in bindings:
                if limit and total_count >= limit:
                    break

                record, skip_reason = self._parse_bulk_binding(
                    binding, person_type=person_type
                )
                if record is None:
                    skipped_count += 1
                    if skip_reason:
                        logger.debug(f"Skipped: {skip_reason}")
                    continue

                # Deduplicate by QID
                if record.source_id in seen_qids:
                    skipped_count += 1
                    continue

                seen_qids.add(record.source_id)
                total_count += 1
                batch_count += 1
                yield record

            logger.info(
                f"Processed {batch_count} people, skipped {skipped_count} (total: {total_count})"
            )

            if len(bindings) < retry_batch_size:
                # Last batch
                break

            offset += retry_batch_size

            # Be polite to the endpoint
            if self._delay > 0:
                time.sleep(self._delay)

        logger.info(f"Completed Wikidata bulk import: {total_count} records (use enrich to add role/org)")

    def _import_all_types(self, limit: Optional[int]) -> Iterator[PersonRecord]:
        """Import from all query types sequentially, deduplicating across types."""
        # Track seen QIDs across all types
        seen_qids: set[str] = set()
        total_count = 0

        # Calculate per-type limits if a total limit is set
        num_types = len(BULK_QUERY_TYPES)
        per_type_limit = limit // num_types if limit else None

        for query_type in BULK_QUERY_TYPES:
            logger.info(f"=== Importing people: {query_type} ===")
            type_count = 0
            skipped_count = 0

            for record in self.import_from_sparql(limit=per_type_limit, query_type=query_type):
                if record.source_id in seen_qids:
                    skipped_count += 1
                    continue

                seen_qids.add(record.source_id)
                total_count += 1
                type_count += 1
                yield record

                if limit and total_count >= limit:
                    logger.info(f"Reached total limit of {limit} records")
                    return

            logger.info(
                f"Got {type_count} new from {query_type}, skipped {skipped_count} (total: {total_count})"
            )

        logger.info(f"Completed all query types: {total_count} total people records")

    @staticmethod
    def _parse_wikidata_date(date_str: str) -> Optional[str]:
        """
        Parse a Wikidata date string into ISO format (YYYY-MM-DD).

        Wikidata returns dates like "2020-01-15T00:00:00Z" or just "2020".
        Returns None if the date cannot be parsed.
        """
        if not date_str:
            return None
        # Handle ISO datetime format (e.g., "2020-01-15T00:00:00Z")
        if "T" in date_str:
            return date_str.split("T")[0]
        # Handle year-only format (e.g., "2020")
        if len(date_str) == 4 and date_str.isdigit():
            return f"{date_str}-01-01"
        # Return as-is if it looks like a date
        if len(date_str) >= 4:
            return date_str[:10]  # Take first 10 chars (YYYY-MM-DD)
        return None

    def _execute_sparql(self, query: str) -> dict[str, Any]:
        """Execute a SPARQL query against Wikidata."""
        params = urllib.parse.urlencode({
            "query": query,
            "format": "json",
        })

        url = f"{WIKIDATA_SPARQL_URL}?{params}"

        req = urllib.request.Request(
            url,
            headers={
                "Accept": "application/sparql-results+json",
                "User-Agent": "corp-extractor/1.0 (person database builder)",
            }
        )

        with urllib.request.urlopen(req, timeout=self._timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def _parse_bulk_binding(
        self,
        binding: dict[str, Any],
        person_type: PersonType = PersonType.UNKNOWN,
    ) -> tuple[Optional[PersonRecord], Optional[str]]:
        """
        Parse a bulk SPARQL result binding into a PersonRecord.

        Bulk bindings only have: person, personLabel, countryLabel, description.
        Role/org/dates are NOT included - use enrich methods to add them later.

        Returns:
            Tuple of (PersonRecord or None, skip_reason or None)
        """
        try:
            # Get Wikidata entity ID
            person_uri = binding.get("person", {}).get("value", "")
            if not person_uri:
                return None, "missing person URI"

            # Extract QID from URI (e.g., "http://www.wikidata.org/entity/Q312" -> "Q312")
            wikidata_id = person_uri.split("/")[-1]
            if not wikidata_id.startswith("Q"):
                return None, f"invalid Wikidata ID format: {wikidata_id}"

            # Get label
            label = binding.get("personLabel", {}).get("value", "")
            if not label:
                return None, f"{wikidata_id}: no label"
            if label == wikidata_id:
                return None, f"{wikidata_id}: no English label (label equals QID)"

            # Get optional fields from bulk query
            country = binding.get("countryLabel", {}).get("value", "")
            description = binding.get("description", {}).get("value", "")

            # Build minimal record data
            record_data: dict[str, Any] = {
                "wikidata_id": wikidata_id,
                "label": label,
            }
            if country:
                record_data["country"] = country
            if description:
                record_data["description"] = description

            return PersonRecord(
                name=label.strip(),
                source="wikidata",
                source_id=wikidata_id,
                country=country or "",
                person_type=person_type,
                known_for_role="",  # To be enriched later
                known_for_org="",   # To be enriched later
                from_date=None,     # To be enriched later
                to_date=None,       # To be enriched later
                record=record_data,
            ), None

        except Exception as e:
            return None, f"parse error: {e}"

    def _parse_binding_with_reason(
        self,
        binding: dict[str, Any],
        person_type: PersonType = PersonType.UNKNOWN,
    ) -> tuple[Optional[PersonRecord], Optional[str]]:
        """
        Parse a SPARQL result binding into a PersonRecord.

        Returns:
            Tuple of (PersonRecord or None, skip_reason or None)
        """
        try:
            # Get Wikidata entity ID
            person_uri = binding.get("person", {}).get("value", "")
            if not person_uri:
                return None, "missing person URI"

            # Extract QID from URI (e.g., "http://www.wikidata.org/entity/Q312" -> "Q312")
            wikidata_id = person_uri.split("/")[-1]
            if not wikidata_id.startswith("Q"):
                return None, f"invalid Wikidata ID format: {wikidata_id}"

            # Get label
            label = binding.get("personLabel", {}).get("value", "")
            if not label:
                return None, f"{wikidata_id}: no label"
            if label == wikidata_id:
                return None, f"{wikidata_id}: no English label (label equals QID)"

            # Get optional fields
            country = binding.get("countryLabel", {}).get("value", "")
            role = binding.get("roleLabel", {}).get("value", "")
            org_label = binding.get("orgLabel", {}).get("value", "")
            org_uri = binding.get("org", {}).get("value", "")
            description = binding.get("description", {}).get("value", "")

            # Extract org QID from URI (e.g., "http://www.wikidata.org/entity/Q715583" -> "Q715583")
            org_qid = ""
            if org_uri:
                org_qid = org_uri.split("/")[-1]
                if not org_qid.startswith("Q"):
                    org_qid = ""

            # Get dates (Wikidata returns ISO datetime, extract just the date part)
            start_date_raw = binding.get("startDate", {}).get("value", "")
            end_date_raw = binding.get("endDate", {}).get("value", "")
            from_date = WikidataPeopleImporter._parse_wikidata_date(start_date_raw)
            to_date = WikidataPeopleImporter._parse_wikidata_date(end_date_raw)

            # Clean up role and org label (remove QID if it's the same as the label)
            if role and role.startswith("Q"):
                role = ""
            if org_label and org_label.startswith("Q"):
                org_label = ""

            # Track discovered organization if we have both QID and label
            if org_qid and org_label:
                self._discovered_orgs[org_qid] = org_label

            # Build record data
            record_data: dict[str, Any] = {
                "wikidata_id": wikidata_id,
                "label": label,
            }
            if country:
                record_data["country"] = country
            if role:
                record_data["role"] = role
            if org_label:
                record_data["org"] = org_label
            if org_qid:
                record_data["org_qid"] = org_qid
            if description:
                record_data["description"] = description
            if from_date:
                record_data["from_date"] = from_date
            if to_date:
                record_data["to_date"] = to_date

            return PersonRecord(
                name=label.strip(),
                source="wikidata",
                source_id=wikidata_id,
                country=country or "",
                person_type=person_type,
                known_for_role=role or "",
                known_for_org=org_label or "",
                from_date=from_date,
                to_date=to_date,
                record=record_data,
            ), None

        except Exception as e:
            return None, f"parse error: {e}"

    def _parse_binding(
        self,
        binding: dict[str, Any],
        person_type: PersonType = PersonType.UNKNOWN,
    ) -> Optional[PersonRecord]:
        """Parse a SPARQL result binding into a PersonRecord (legacy wrapper)."""
        record, _ = self._parse_binding_with_reason(binding, person_type)
        return record

    def search_person(self, name: str, limit: int = 10) -> list[PersonRecord]:
        """
        Search for a specific person by name.

        Args:
            name: Person name to search for
            limit: Maximum results to return

        Returns:
            List of matching PersonRecords
        """
        # Use Wikidata search API for better name matching
        search_url = "https://www.wikidata.org/w/api.php"
        params = urllib.parse.urlencode({
            "action": "wbsearchentities",
            "search": name,
            "language": "en",
            "type": "item",
            "limit": limit,
            "format": "json",
        })

        req = urllib.request.Request(
            f"{search_url}?{params}",
            headers={"User-Agent": "corp-extractor/1.0"}
        )

        with urllib.request.urlopen(req, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))

        results = []
        for item in data.get("search", []):
            qid = item.get("id")
            label = item.get("label", "")
            description = item.get("description", "")

            # Check if it looks like a person
            person_keywords = [
                "politician", "actor", "actress", "singer", "musician",
                "businessman", "businesswoman", "ceo", "executive", "director",
                "president", "founder", "professor", "scientist", "author",
                "writer", "journalist", "athlete", "player", "coach",
            ]
            description_lower = description.lower()
            is_person = any(kw in description_lower for kw in person_keywords)
            if not is_person:
                continue

            # Try to infer person type from description
            person_type = PersonType.UNKNOWN
            if any(kw in description_lower for kw in ["ceo", "executive", "businessman", "businesswoman"]):
                person_type = PersonType.EXECUTIVE
            elif any(kw in description_lower for kw in ["politician", "president", "senator", "minister"]):
                person_type = PersonType.POLITICIAN
            elif any(kw in description_lower for kw in ["athlete", "player", "coach"]):
                person_type = PersonType.ATHLETE
            elif any(kw in description_lower for kw in ["actor", "actress", "singer", "musician", "director"]):
                person_type = PersonType.ARTIST
            elif any(kw in description_lower for kw in ["professor", "academic"]):
                person_type = PersonType.ACADEMIC
            elif any(kw in description_lower for kw in ["scientist", "researcher"]):
                person_type = PersonType.SCIENTIST
            elif any(kw in description_lower for kw in ["journalist", "reporter"]):
                person_type = PersonType.JOURNALIST
            elif any(kw in description_lower for kw in ["founder", "entrepreneur"]):
                person_type = PersonType.ENTREPRENEUR

            record = PersonRecord(
                name=label,
                source="wikidata",
                source_id=qid,
                country="",  # Not available from search API
                person_type=person_type,
                known_for_role="",
                known_for_org="",
                record={
                    "wikidata_id": qid,
                    "label": label,
                    "description": description,
                },
            )
            results.append(record)

        return results

    def get_discovered_organizations(self) -> list[CompanyRecord]:
        """
        Get organizations discovered during the people import.

        These are organizations associated with people (employers, positions, etc.)
        that can be inserted into the organizations database if not already present.

        Returns:
            List of CompanyRecord objects for discovered organizations
        """
        records = []
        for org_qid, org_label in self._discovered_orgs.items():
            record = CompanyRecord(
                name=org_label,
                source="wikipedia",  # Use "wikipedia" as source per wikidata.py convention
                source_id=org_qid,
                region="",  # Not available from this context
                entity_type=EntityType.BUSINESS,  # Default to business for orgs linked to people
                record={
                    "wikidata_id": org_qid,
                    "label": org_label,
                    "discovered_from": "people_import",
                },
            )
            records.append(record)
        logger.info(f"Discovered {len(records)} organizations from people import")
        return records

    def clear_discovered_organizations(self) -> None:
        """Clear the discovered organizations cache."""
        self._discovered_orgs.clear()

    def enrich_person_dates(self, person_qid: str, role: str = "", org: str = "") -> tuple[Optional[str], Optional[str]]:
        """
        Query Wikidata to get start/end dates for a person's position.

        Args:
            person_qid: Wikidata QID of the person (e.g., 'Q123')
            role: Optional role label to match (e.g., 'chief executive officer')
            org: Optional org label to match (e.g., 'Apple Inc')

        Returns:
            Tuple of (from_date, to_date) in ISO format, or (None, None) if not found
        """
        # Query for position dates for this specific person
        query = """
        SELECT ?roleLabel ?orgLabel ?startDate ?endDate WHERE {
          wd:%s p:P39 ?positionStatement .
          ?positionStatement ps:P39 ?role .
          OPTIONAL { ?positionStatement pq:P642 ?org }
          OPTIONAL { ?positionStatement pq:P580 ?startDate }
          OPTIONAL { ?positionStatement pq:P582 ?endDate }
          SERVICE wikibase:label { bd:serviceParam wikibase:language "en,mul". }
        }
        LIMIT 50
        """ % person_qid

        try:
            url = f"{WIKIDATA_SPARQL_URL}?query={urllib.parse.quote(query)}&format=json"
            req = urllib.request.Request(url, headers={"User-Agent": "corp-extractor/1.0"})

            with urllib.request.urlopen(req, timeout=30) as response:
                data = json.loads(response.read().decode("utf-8"))

            # Find the best matching position
            best_start = None
            best_end = None

            for binding in data.get("results", {}).get("bindings", []):
                role_label = binding.get("roleLabel", {}).get("value", "")
                org_label = binding.get("orgLabel", {}).get("value", "")
                start_raw = binding.get("startDate", {}).get("value", "")
                end_raw = binding.get("endDate", {}).get("value", "")

                # If role/org specified, try to match
                if role and role.lower() not in role_label.lower():
                    continue
                if org and org.lower() not in org_label.lower():
                    continue

                # Parse dates
                start_date = self._parse_wikidata_date(start_raw)
                end_date = self._parse_wikidata_date(end_raw)

                # Prefer entries with dates
                if start_date or end_date:
                    best_start = start_date
                    best_end = end_date
                    break  # Found a match with dates

            return best_start, best_end

        except Exception as e:
            logger.debug(f"Failed to enrich dates for {person_qid}: {e}")
            return None, None

    def enrich_people_batch(
        self,
        people: list[PersonRecord],
        delay_seconds: float = 0.5,
    ) -> int:
        """
        Enrich a batch of people with start/end dates.

        Args:
            people: List of PersonRecord objects to enrich
            delay_seconds: Delay between requests

        Returns:
            Number of people enriched with dates
        """
        enriched_count = 0

        for person in people:
            if person.from_date or person.to_date:
                continue  # Already has dates

            qid = person.source_id
            role = person.known_for_role
            org = person.known_for_org

            from_date, to_date = self.enrich_person_dates(qid, role, org)

            if from_date or to_date:
                person.from_date = from_date
                person.to_date = to_date
                enriched_count += 1
                logger.debug(f"Enriched {person.name}: {from_date} - {to_date}")

            time.sleep(delay_seconds)

        logger.info(f"Enriched {enriched_count}/{len(people)} people with dates")
        return enriched_count

    def enrich_person_role_org(
        self, person_qid: str
    ) -> tuple[str, str, str, Optional[str], Optional[str]]:
        """
        Query Wikidata to get role, org, and dates for a person.

        Args:
            person_qid: Wikidata QID of the person (e.g., 'Q123')

        Returns:
            Tuple of (role_label, org_label, org_qid, from_date, to_date)
            Empty strings/None if not found
        """
        # Query for position held (P39) with org qualifier and dates
        query = """
        SELECT ?roleLabel ?org ?orgLabel ?startDate ?endDate WHERE {
          wd:%s p:P39 ?stmt .
          ?stmt ps:P39 ?role .
          OPTIONAL { ?stmt pq:P642 ?org . }
          OPTIONAL { ?stmt pq:P580 ?startDate . }
          OPTIONAL { ?stmt pq:P582 ?endDate . }
          SERVICE wikibase:label { bd:serviceParam wikibase:language "en,mul". }
        }
        LIMIT 5
        """ % person_qid

        try:
            url = f"{WIKIDATA_SPARQL_URL}?query={urllib.parse.quote(query)}&format=json"
            req = urllib.request.Request(url, headers={"User-Agent": "corp-extractor/1.0"})

            with urllib.request.urlopen(req, timeout=30) as response:
                data = json.loads(response.read().decode("utf-8"))

            bindings = data.get("results", {}).get("bindings", [])

            # Find the best result (prefer one with org and dates)
            best_result = None
            for binding in bindings:
                role_label = binding.get("roleLabel", {}).get("value", "")
                org_label = binding.get("orgLabel", {}).get("value", "")
                org_uri = binding.get("org", {}).get("value", "")
                start_raw = binding.get("startDate", {}).get("value", "")
                end_raw = binding.get("endDate", {}).get("value", "")

                # Skip if role is just a QID (no label resolved)
                if role_label and role_label.startswith("Q"):
                    continue
                if org_label and org_label.startswith("Q"):
                    org_label = ""

                # Extract QID from URI
                org_qid = ""
                if org_uri:
                    org_qid = org_uri.split("/")[-1]
                    if not org_qid.startswith("Q"):
                        org_qid = ""

                from_date = self._parse_wikidata_date(start_raw)
                to_date = self._parse_wikidata_date(end_raw)

                result = (role_label, org_label, org_qid, from_date, to_date)

                # Prefer results with org and dates
                if org_label and (from_date or to_date):
                    return result
                elif org_label and best_result is None:
                    best_result = result
                elif role_label and best_result is None:
                    best_result = result

            if best_result:
                return best_result

            return "", "", "", None, None

        except Exception as e:
            logger.debug(f"Failed to enrich role/org for {person_qid}: {e}")
            return "", "", "", None, None

    def enrich_people_role_org_batch(
        self,
        people: list[PersonRecord],
        delay_seconds: float = 0.1,
        max_workers: int = 5,
    ) -> int:
        """
        Enrich a batch of people with role/org/dates data using parallel queries.

        Args:
            people: List of PersonRecord objects to enrich
            delay_seconds: Delay between requests (per worker)
            max_workers: Number of parallel workers (default 5 for Wikidata rate limits)

        Returns:
            Number of people enriched with role/org
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # Filter to people that need enrichment
        to_enrich = [p for p in people if not p.known_for_role and not p.known_for_org]

        if not to_enrich:
            logger.info("No people need enrichment")
            return 0

        enriched_count = 0
        total = len(to_enrich)

        def enrich_one(person: PersonRecord) -> tuple[PersonRecord, bool]:
            """Enrich a single person, returns (person, success)."""
            try:
                role, org, org_qid, from_date, to_date = self.enrich_person_role_org(person.source_id)

                if role or org:
                    person.known_for_role = role
                    person.known_for_org = org
                    if org_qid:
                        person.record["org_qid"] = org_qid
                    if from_date:
                        person.from_date = from_date
                    if to_date:
                        person.to_date = to_date
                    return person, True

                return person, False
            except Exception as e:
                logger.debug(f"Failed to enrich {person.source_id}: {e}")
                return person, False

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            futures = {executor.submit(enrich_one, person): person for person in to_enrich}

            # Process results as they complete
            completed = 0
            for future in as_completed(futures):
                person, success = future.result()
                if success:
                    enriched_count += 1
                    logger.debug(f"Enriched {person.name}: {person.known_for_role} at {person.known_for_org}")

                completed += 1
                if completed % 100 == 0:
                    logger.info(f"Enriched {completed}/{total} people ({enriched_count} with data)...")

                # Small delay to avoid rate limiting
                time.sleep(delay_seconds)

        logger.info(f"Enriched {enriched_count}/{total} people with role/org/dates")
        return enriched_count

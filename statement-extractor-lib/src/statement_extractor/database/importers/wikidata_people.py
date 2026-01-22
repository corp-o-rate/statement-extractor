"""
Wikidata importer for the person database.

Imports notable people data from Wikidata using SPARQL queries
into the embedding database for person name matching.

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

from ..models import PersonRecord, PersonType

logger = logging.getLogger(__name__)

# Wikidata SPARQL endpoint
WIKIDATA_SPARQL_URL = "https://query.wikidata.org/sparql"

# Base query template for people with Wikipedia articles
# Gets person, their position/role, and organization
PERSON_BASE_QUERY = """
SELECT DISTINCT ?person ?personLabel ?countryLabel ?roleLabel ?orgLabel ?description WHERE {{
  ?person wdt:P31 wd:Q5 .  # Instance of human

  # Filter condition specific to query type
  {filter_condition}

  # Get country of citizenship
  OPTIONAL {{ ?person wdt:P27 ?country. }}

  # Get position held and associated organization
  OPTIONAL {{
    ?person p:P39 ?positionStatement .
    ?positionStatement ps:P39 ?role .
    OPTIONAL {{ ?positionStatement pq:P642 ?org }}  # "of" qualifier
  }}

  # Fallback: direct employer
  OPTIONAL {{ ?person wdt:P108 ?employer. BIND(?employer AS ?org) }}

  # Get description
  OPTIONAL {{ ?person schema:description ?description FILTER(LANG(?description) = "en") }}

  # Must have English Wikipedia article (notability filter)
  ?article schema:about ?person ; schema:isPartOf <https://en.wikipedia.org/> .

  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
}}
LIMIT %d
OFFSET %d
"""

# Query for business executives (CEOs, CFOs, board members, etc.) - P39 = executive positions
EXECUTIVE_QUERY = """
SELECT DISTINCT ?person ?personLabel ?countryLabel ?roleLabel ?orgLabel ?description WHERE {
  ?person wdt:P31 wd:Q5 .  # Instance of human

  # Has held executive position
  ?person p:P39 ?positionStatement .
  ?positionStatement ps:P39 ?role .

  # Role is a type of corporate officer, board member, or executive
  VALUES ?role {
    # C-Suite
    wd:Q484876    # CEO (Chief Executive Officer)
    wd:Q623279    # CFO (Chief Financial Officer)
    wd:Q1502675   # CTO (Chief Technology Officer)
    wd:Q935019    # COO (Chief Operating Officer)
    wd:Q1057716   # CMO (Chief Marketing Officer)
    wd:Q2140589   # CIO (Chief Information Officer)
    wd:Q1115042   # Chief Human Resources Officer
    wd:Q4720025   # Chief Legal Officer / General Counsel
    wd:Q60432825  # Chief Product Officer
    wd:Q15967139  # Chief Strategy Officer
    wd:Q15729310  # Chief Revenue Officer
    wd:Q47523568  # Chief Digital Officer

    # Board positions
    wd:Q258557    # Chairman / Chairman of the Board
    wd:Q114863313 # Vice Chairman
    wd:Q726114    # President (business)
    wd:Q1372944   # Vice President
    wd:Q18918145  # Executive Vice President
    wd:Q1057569   # Board of directors member
    wd:Q24058752  # Non-executive director
    wd:Q3578048   # Independent director

    # Other executive roles
    wd:Q476675    # Managing Director
    wd:Q5441744   # Executive Director
    wd:Q4188234   # General Manager
    wd:Q38844673  # Group CEO
    wd:Q97273203  # President and CEO
    wd:Q60715311  # Chairman and CEO
    wd:Q3563879   # Partner (business)
    wd:Q3505845   # Senior Partner
  }

  OPTIONAL { ?positionStatement pq:P642 ?org }  # "of" qualifier
  OPTIONAL { ?person wdt:P108 ?employer. BIND(?employer AS ?org) }
  OPTIONAL { ?person wdt:P27 ?country. }
  OPTIONAL { ?person schema:description ?description FILTER(LANG(?description) = "en") }

  ?article schema:about ?person ; schema:isPartOf <https://en.wikipedia.org/> .
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
LIMIT %d
OFFSET %d
"""

# Query for politicians
POLITICIAN_QUERY = """
SELECT DISTINCT ?person ?personLabel ?countryLabel ?roleLabel ?orgLabel ?description WHERE {
  ?person wdt:P31 wd:Q5 .  # Instance of human

  # Occupation is politician
  ?person wdt:P106 wd:Q82955 .

  OPTIONAL {
    ?person p:P39 ?positionStatement .
    ?positionStatement ps:P39 ?role .
    OPTIONAL { ?positionStatement pq:P642 ?org }
  }
  OPTIONAL { ?person wdt:P27 ?country. }
  OPTIONAL { ?person schema:description ?description FILTER(LANG(?description) = "en") }

  ?article schema:about ?person ; schema:isPartOf <https://en.wikipedia.org/> .
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
LIMIT %d
OFFSET %d
"""

# Query for athletes
ATHLETE_QUERY = """
SELECT DISTINCT ?person ?personLabel ?countryLabel ?roleLabel ?orgLabel ?description WHERE {
  ?person wdt:P31 wd:Q5 .  # Instance of human

  # Is an athlete (has sports team membership P54 or is athlete P106)
  { ?person wdt:P106 wd:Q2066131 . }  # Athlete occupation
  UNION
  { ?person wdt:P54 ?team . }  # Member of sports team

  OPTIONAL { ?person wdt:P54 ?team . BIND(?team AS ?org) }
  OPTIONAL { ?person wdt:P27 ?country. }
  OPTIONAL { ?person schema:description ?description FILTER(LANG(?description) = "en") }

  ?article schema:about ?person ; schema:isPartOf <https://en.wikipedia.org/> .
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
LIMIT %d
OFFSET %d
"""

# Query for artists (actors, musicians, directors)
ARTIST_QUERY = """
SELECT DISTINCT ?person ?personLabel ?countryLabel ?roleLabel ?orgLabel ?description WHERE {
  ?person wdt:P31 wd:Q5 .  # Instance of human

  # Has artist occupation
  ?person wdt:P106 ?occupation .
  VALUES ?occupation {
    wd:Q33999    # Actor
    wd:Q177220   # Singer
    wd:Q639669   # Musician
    wd:Q2526255  # Film director
    wd:Q36180    # Writer
    wd:Q483501   # Artist
  }

  OPTIONAL { ?person wdt:P108 ?employer. BIND(?employer AS ?org) }
  OPTIONAL { ?person wdt:P27 ?country. }
  OPTIONAL { ?person schema:description ?description FILTER(LANG(?description) = "en") }

  ?article schema:about ?person ; schema:isPartOf <https://en.wikipedia.org/> .
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
LIMIT %d
OFFSET %d
"""

# Query for academics (professors)
ACADEMIC_QUERY = """
SELECT DISTINCT ?person ?personLabel ?countryLabel ?roleLabel ?orgLabel ?description WHERE {
  ?person wdt:P31 wd:Q5 .  # Instance of human

  # Is a professor or academic
  { ?person wdt:P106 wd:Q121594 . }  # Professor
  UNION
  { ?person wdt:P106 wd:Q3400985 . }  # Academic

  OPTIONAL { ?person wdt:P108 ?employer. BIND(?employer AS ?org) }
  OPTIONAL { ?person wdt:P27 ?country. }
  OPTIONAL { ?person schema:description ?description FILTER(LANG(?description) = "en") }

  ?article schema:about ?person ; schema:isPartOf <https://en.wikipedia.org/> .
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
LIMIT %d
OFFSET %d
"""

# Query for scientists
SCIENTIST_QUERY = """
SELECT DISTINCT ?person ?personLabel ?countryLabel ?roleLabel ?orgLabel ?description WHERE {
  ?person wdt:P31 wd:Q5 .  # Instance of human

  # Is a scientist or researcher
  { ?person wdt:P106 wd:Q901 . }  # Scientist
  UNION
  { ?person wdt:P106 wd:Q1650915 . }  # Researcher

  OPTIONAL { ?person wdt:P108 ?employer. BIND(?employer AS ?org) }
  OPTIONAL { ?person wdt:P27 ?country. }
  OPTIONAL { ?person schema:description ?description FILTER(LANG(?description) = "en") }

  ?article schema:about ?person ; schema:isPartOf <https://en.wikipedia.org/> .
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
LIMIT %d
OFFSET %d
"""

# Query for journalists and media personalities
JOURNALIST_QUERY = """
SELECT DISTINCT ?person ?personLabel ?countryLabel ?roleLabel ?orgLabel ?description WHERE {
  ?person wdt:P31 wd:Q5 .  # Instance of human

  # Is a journalist or presenter
  { ?person wdt:P106 wd:Q1930187 . }  # Journalist
  UNION
  { ?person wdt:P106 wd:Q13590141 . }  # Television presenter

  OPTIONAL { ?person wdt:P108 ?employer. BIND(?employer AS ?org) }
  OPTIONAL { ?person wdt:P27 ?country. }
  OPTIONAL { ?person schema:description ?description FILTER(LANG(?description) = "en") }

  ?article schema:about ?person ; schema:isPartOf <https://en.wikipedia.org/> .
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
LIMIT %d
OFFSET %d
"""

# Query for entrepreneurs (founders)
ENTREPRENEUR_QUERY = """
SELECT DISTINCT ?person ?personLabel ?countryLabel ?roleLabel ?orgLabel ?description WHERE {
  ?person wdt:P31 wd:Q5 .  # Instance of human

  # Founded a company (inverse of P112)
  ?org wdt:P112 ?person .

  OPTIONAL { ?person wdt:P27 ?country. }
  OPTIONAL { ?person schema:description ?description FILTER(LANG(?description) = "en") }

  ?article schema:about ?person ; schema:isPartOf <https://en.wikipedia.org/> .
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
LIMIT %d
OFFSET %d
"""

# Query for activists
ACTIVIST_QUERY = """
SELECT DISTINCT ?person ?personLabel ?countryLabel ?roleLabel ?orgLabel ?description WHERE {
  ?person wdt:P31 wd:Q5 .  # Instance of human

  # Is an activist
  ?person wdt:P106 wd:Q15253558 .  # Activist

  OPTIONAL { ?person wdt:P108 ?employer. BIND(?employer AS ?org) }
  OPTIONAL { ?person wdt:P27 ?country. }
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

# Mapping query type to SPARQL query template
QUERY_TYPES: dict[str, str] = {
    "executive": EXECUTIVE_QUERY,
    "politician": POLITICIAN_QUERY,
    "athlete": ATHLETE_QUERY,
    "artist": ARTIST_QUERY,
    "academic": ACADEMIC_QUERY,
    "scientist": SCIENTIST_QUERY,
    "journalist": JOURNALIST_QUERY,
    "entrepreneur": ENTREPRENEUR_QUERY,
    "activist": ACTIVIST_QUERY,
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

    def __init__(self, batch_size: int = 1000, delay_seconds: float = 2.0, timeout: int = 120):
        """
        Initialize the Wikidata people importer.

        Args:
            batch_size: Number of records to fetch per SPARQL query (default 1000)
            delay_seconds: Delay between requests to be polite to the endpoint
            timeout: HTTP timeout in seconds (default 120)
        """
        self._batch_size = batch_size
        self._delay = delay_seconds
        self._timeout = timeout

    def import_from_sparql(
        self,
        limit: Optional[int] = None,
        query_type: str = "executive",
        import_all: bool = False,
    ) -> Iterator[PersonRecord]:
        """
        Import person records from Wikidata via SPARQL.

        Args:
            limit: Optional limit on total records
            query_type: Which query to use (executive, politician, athlete, etc.)
            import_all: If True, run all query types sequentially

        Yields:
            PersonRecord for each person
        """
        if import_all:
            yield from self._import_all_types(limit)
            return

        if query_type not in QUERY_TYPES:
            raise ValueError(f"Unknown query type: {query_type}. Use one of: {list(QUERY_TYPES.keys())}")

        query_template = QUERY_TYPES[query_type]
        person_type = QUERY_TYPE_TO_PERSON_TYPE.get(query_type, PersonType.UNKNOWN)
        logger.info(f"Starting Wikidata people import via SPARQL (query_type={query_type}, person_type={person_type.value})...")

        offset = 0
        total_count = 0
        seen_ids = set()  # Track seen Wikidata IDs to avoid duplicates

        while True:
            if limit and total_count >= limit:
                break

            batch_limit = min(self._batch_size, (limit - total_count) if limit else self._batch_size)
            query = query_template % (batch_limit, offset)

            logger.info(f"Fetching Wikidata people batch at offset {offset}...")

            try:
                results = self._execute_sparql(query)
            except Exception as e:
                logger.error(f"SPARQL query failed at offset {offset}: {e}")
                break

            bindings = results.get("results", {}).get("bindings", [])

            if not bindings:
                logger.info("No more results from Wikidata")
                break

            batch_count = 0
            for binding in bindings:
                if limit and total_count >= limit:
                    break

                record = self._parse_binding(binding, person_type=person_type)
                if record and record.source_id not in seen_ids:
                    seen_ids.add(record.source_id)
                    total_count += 1
                    batch_count += 1
                    yield record

            logger.info(f"Processed {batch_count} people from batch (total: {total_count})")

            if len(bindings) < batch_limit:
                # Last batch
                break

            offset += self._batch_size

            # Be polite to the endpoint
            if self._delay > 0:
                time.sleep(self._delay)

        logger.info(f"Completed Wikidata people import: {total_count} records")

    def _import_all_types(self, limit: Optional[int]) -> Iterator[PersonRecord]:
        """Import from all query types sequentially, deduplicating across types."""
        seen_ids: set[str] = set()
        total_count = 0

        # Calculate per-type limits if a total limit is set
        num_types = len(QUERY_TYPES)
        per_type_limit = limit // num_types if limit else None

        for query_type in QUERY_TYPES:
            logger.info(f"=== Importing people: {query_type} ===")
            type_count = 0

            for record in self.import_from_sparql(limit=per_type_limit, query_type=query_type):
                if record.source_id not in seen_ids:
                    seen_ids.add(record.source_id)
                    total_count += 1
                    type_count += 1
                    yield record

                    if limit and total_count >= limit:
                        logger.info(f"Reached total limit of {limit} records")
                        return

            logger.info(f"Got {type_count} new records from {query_type} (total: {total_count})")

        logger.info(f"Completed all query types: {total_count} total people records")

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

    def _parse_binding(
        self,
        binding: dict[str, Any],
        person_type: PersonType = PersonType.UNKNOWN,
    ) -> Optional[PersonRecord]:
        """Parse a SPARQL result binding into a PersonRecord."""
        try:
            # Get Wikidata entity ID
            person_uri = binding.get("person", {}).get("value", "")
            if not person_uri:
                return None

            # Extract QID from URI (e.g., "http://www.wikidata.org/entity/Q312" -> "Q312")
            wikidata_id = person_uri.split("/")[-1]
            if not wikidata_id.startswith("Q"):
                return None

            # Get label
            label = binding.get("personLabel", {}).get("value", "")
            if not label or label == wikidata_id:  # Skip if no English label
                return None

            # Get optional fields
            country = binding.get("countryLabel", {}).get("value", "")
            role = binding.get("roleLabel", {}).get("value", "")
            org = binding.get("orgLabel", {}).get("value", "")
            description = binding.get("description", {}).get("value", "")

            # Clean up role and org (remove QID if it's the same as the label)
            if role and role.startswith("Q"):
                role = ""
            if org and org.startswith("Q"):
                org = ""

            # Build record data
            record_data: dict[str, Any] = {
                "wikidata_id": wikidata_id,
                "label": label,
            }
            if country:
                record_data["country"] = country
            if role:
                record_data["role"] = role
            if org:
                record_data["org"] = org
            if description:
                record_data["description"] = description

            return PersonRecord(
                name=label.strip(),
                source="wikidata",
                source_id=wikidata_id,
                country=country or "",
                person_type=person_type,
                known_for_role=role or "",
                known_for_org=org or "",
                record=record_data,
            )

        except Exception as e:
            logger.debug(f"Failed to parse Wikidata binding: {e}")
            return None

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

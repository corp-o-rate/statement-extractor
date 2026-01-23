"""
Wikidata dump importer for people and organizations.

Uses the Wikidata JSON dump (~100GB compressed) to import:
1. People: All humans (P31=Q5) with English Wikipedia articles
2. Organizations: All organizations with English Wikipedia articles

This avoids SPARQL query timeouts that occur with large result sets.
The dump is processed line-by-line to minimize memory usage.

Dump format:
- File: `latest-all.json.bz2` (~100GB) or `.gz` (~150GB)
- Format: JSON array where each line is a separate entity (after first `[` line)
- Each line: `{"type":"item","id":"Q123","labels":{...},"claims":{...},"sitelinks":{...}},`
- Streaming: Read line-by-line, strip trailing comma, parse JSON
"""

import bz2
import gzip
import json
import logging
import shutil
import subprocess
import urllib.request
from pathlib import Path
from typing import Callable, Iterator, Optional

from ..models import CompanyRecord, EntityType, PersonRecord, PersonType

logger = logging.getLogger(__name__)

# Wikidata dump URLs - mirrors for faster downloads
# Primary is Wikimedia (slow), alternatives may be faster
DUMP_MIRRORS = [
    # Wikimedia Foundation (official, often slow)
    "https://dumps.wikimedia.org/wikidatawiki/entities/latest-all.json.bz2",
    # Academic Torrents mirror (if available) - typically faster
    # Note: Check https://academictorrents.com/browse?search=wikidata for current links
]

# Default URL (can be overridden)
DUMP_URL = DUMP_MIRRORS[0]

# For even faster downloads, users can:
# 1. Use a torrent client with the Academic Torrents magnet link
# 2. Download from a regional Wikimedia mirror
# 3. Use aria2c with multiple connections: aria2c -x 16 -s 16 <url>

# =============================================================================
# POSITION TO PERSON TYPE MAPPING (P39 - position held)
# =============================================================================

# Executive positions (P39 values)
EXECUTIVE_POSITION_QIDS = {
    "Q484876",    # CEO
    "Q623279",    # CFO
    "Q1502675",   # COO
    "Q935019",    # CTO
    "Q1057716",   # CIO
    "Q2140589",   # CMO
    "Q1115042",   # chairperson
    "Q4720025",   # board of directors member
    "Q60432825",  # chief human resources officer
    "Q15967139",  # chief compliance officer
    "Q15729310",  # chief risk officer
    "Q47523568",  # chief legal officer
    "Q258557",    # board chair
    "Q114863313", # chief sustainability officer
    "Q726114",    # company president
    "Q1372944",   # managing director
    "Q18918145",  # chief commercial officer
    "Q1057569",   # chief strategy officer
    "Q24058752",  # chief product officer
    "Q3578048",   # vice president
    "Q476675",    # business executive (generic)
    "Q5441744",   # finance director
    "Q4188234",   # general manager
    "Q38844673",  # chief data officer
    "Q97273203",  # chief digital officer
    "Q60715311",  # chief growth officer
    "Q3563879",   # treasurer
    "Q3505845",   # corporate secretary
}

# Politician positions (P39 values)
POLITICIAN_POSITION_QIDS = {
    "Q30461",     # president
    "Q14212",     # prime minister
    "Q83307",     # minister
    "Q2285706",   # head of government
    "Q4175034",   # legislator
    "Q486839",    # member of parliament
    "Q193391",    # member of national legislature
    "Q212071",    # mayor
    "Q382617",    # governor
    "Q116",       # monarch
    "Q484529",    # member of congress
    "Q294414",    # public office (generic)
}

# =============================================================================
# OCCUPATION TO PERSON TYPE MAPPING (P106 - occupation)
# =============================================================================

OCCUPATION_TO_TYPE: dict[str, PersonType] = {
    # Politicians - catches people like Andy Burnham!
    "Q82955": PersonType.POLITICIAN,     # politician occupation

    # Athletes
    "Q2066131": PersonType.ATHLETE,      # athlete
    "Q937857": PersonType.ATHLETE,       # football player
    "Q3665646": PersonType.ATHLETE,      # basketball player
    "Q10871364": PersonType.ATHLETE,     # baseball player
    "Q19204627": PersonType.ATHLETE,     # ice hockey player
    "Q10843402": PersonType.ATHLETE,     # tennis player
    "Q13381376": PersonType.ATHLETE,     # golfer
    "Q11338576": PersonType.ATHLETE,     # boxer
    "Q10873124": PersonType.ATHLETE,     # swimmer

    # Artists
    "Q33999": PersonType.ARTIST,         # actor
    "Q177220": PersonType.ARTIST,        # singer
    "Q639669": PersonType.ARTIST,        # musician
    "Q2526255": PersonType.ARTIST,       # film director
    "Q36180": PersonType.ARTIST,         # writer
    "Q483501": PersonType.ARTIST,        # artist
    "Q488205": PersonType.ARTIST,        # singer-songwriter
    "Q753110": PersonType.ARTIST,        # songwriter
    "Q2405480": PersonType.ARTIST,       # voice actor
    "Q10800557": PersonType.ARTIST,      # film actor

    # Academics
    "Q121594": PersonType.ACADEMIC,      # professor
    "Q3400985": PersonType.ACADEMIC,     # academic
    "Q1622272": PersonType.ACADEMIC,     # university professor

    # Scientists
    "Q901": PersonType.SCIENTIST,        # scientist
    "Q1650915": PersonType.SCIENTIST,    # researcher
    "Q169470": PersonType.SCIENTIST,     # physicist
    "Q593644": PersonType.SCIENTIST,     # chemist
    "Q864503": PersonType.SCIENTIST,     # biologist
    "Q11063": PersonType.SCIENTIST,      # astronomer

    # Journalists
    "Q1930187": PersonType.JOURNALIST,   # journalist
    "Q13590141": PersonType.JOURNALIST,  # news presenter
    "Q947873": PersonType.JOURNALIST,    # television presenter
    "Q4263842": PersonType.JOURNALIST,   # columnist

    # Activists
    "Q15253558": PersonType.ACTIVIST,    # activist
    "Q11631410": PersonType.ACTIVIST,    # human rights activist
    "Q18939491": PersonType.ACTIVIST,    # environmental activist

    # Entrepreneurs/Executives via occupation
    "Q131524": PersonType.ENTREPRENEUR,  # entrepreneur
    "Q43845": PersonType.ENTREPRENEUR,   # businessperson
}

# =============================================================================
# ORGANIZATION TYPE MAPPING (P31 - instance of)
# =============================================================================

ORG_TYPE_TO_ENTITY_TYPE: dict[str, EntityType] = {
    # Business
    "Q4830453": EntityType.BUSINESS,     # business
    "Q6881511": EntityType.BUSINESS,     # enterprise
    "Q783794": EntityType.BUSINESS,      # company
    "Q891723": EntityType.BUSINESS,      # public company
    "Q167037": EntityType.BUSINESS,      # corporation
    "Q658255": EntityType.BUSINESS,      # subsidiary
    "Q206652": EntityType.BUSINESS,      # conglomerate
    "Q22687": EntityType.BUSINESS,       # bank
    "Q1145276": EntityType.BUSINESS,     # insurance company
    "Q46970": EntityType.BUSINESS,       # airline
    "Q613142": EntityType.BUSINESS,      # law firm
    "Q507619": EntityType.BUSINESS,      # pharmaceutical company
    "Q2979960": EntityType.BUSINESS,     # technology company
    "Q1631111": EntityType.BUSINESS,     # retailer
    "Q187652": EntityType.BUSINESS,      # manufacturer

    # Funds
    "Q45400320": EntityType.FUND,        # investment fund
    "Q476028": EntityType.FUND,          # hedge fund
    "Q380649": EntityType.FUND,          # investment company

    # Nonprofits
    "Q163740": EntityType.NONPROFIT,     # nonprofit organization
    "Q79913": EntityType.NGO,            # non-governmental organization
    "Q157031": EntityType.FOUNDATION,    # foundation

    # Government
    "Q327333": EntityType.GOVERNMENT,    # government agency
    "Q484652": EntityType.INTERNATIONAL_ORG,  # international organization
    "Q7278": EntityType.POLITICAL_PARTY, # political party
    "Q178790": EntityType.TRADE_UNION,   # trade union

    # Education/Research
    "Q2385804": EntityType.EDUCATIONAL,  # educational institution
    "Q3918": EntityType.EDUCATIONAL,     # university
    "Q31855": EntityType.RESEARCH,       # research institute

    # Other
    "Q16917": EntityType.HEALTHCARE,     # hospital
    "Q476028": EntityType.SPORTS,        # sports club (hedge fund duplicate, fix below)
    "Q847017": EntityType.SPORTS,        # sports club
    "Q18127": EntityType.MEDIA,          # record label
    "Q1366047": EntityType.MEDIA,        # film studio
    "Q1137109": EntityType.MEDIA,        # video game company
}


class WikidataDumpImporter:
    """
    Stream Wikidata JSON dump to extract people and organization records.

    This importer processes the Wikidata dump line-by-line to avoid memory issues
    with the ~100GB compressed file. It filters for:
    - Humans (P31=Q5) with English Wikipedia articles
    - Organizations with English Wikipedia articles

    The dump URL can be customized, and the importer supports both .bz2 and .gz
    compression formats.
    """

    def __init__(self, dump_path: Optional[str] = None):
        """
        Initialize the dump importer.

        Args:
            dump_path: Optional path to a pre-downloaded dump file.
                      If not provided, will need to call download_dump() first.
        """
        self._dump_path = Path(dump_path) if dump_path else None
        # Track discovered organizations from people import
        self._discovered_orgs: dict[str, str] = {}

    def download_dump(
        self,
        target_dir: Optional[Path] = None,
        force: bool = False,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        use_aria2: bool = True,
        aria2_connections: int = 16,
    ) -> Path:
        """
        Download the latest Wikidata dump with progress indicator.

        For fastest downloads, uses aria2c if available (16 parallel connections).
        Falls back to urllib if aria2c is not installed.

        Args:
            target_dir: Directory to save the dump (default: ~/.cache/corp-extractor)
            force: Force re-download even if file exists
            progress_callback: Optional callback(downloaded_bytes, total_bytes) for progress
            use_aria2: Try to use aria2c for faster downloads (default: True)
            aria2_connections: Number of connections for aria2c (default: 16)

        Returns:
            Path to the downloaded dump file
        """
        if target_dir is None:
            target_dir = Path.home() / ".cache" / "corp-extractor"

        target_dir.mkdir(parents=True, exist_ok=True)
        dump_path = target_dir / "wikidata-latest-all.json.bz2"

        if dump_path.exists() and not force:
            logger.info(f"Using cached dump at {dump_path}")
            self._dump_path = dump_path
            return dump_path

        logger.info(f"Target: {dump_path}")

        # Try aria2c first for much faster downloads
        if use_aria2 and shutil.which("aria2c"):
            logger.info("Using aria2c for fast parallel download...")
            try:
                self._download_with_aria2(dump_path, connections=aria2_connections)
                self._dump_path = dump_path
                return dump_path
            except Exception as e:
                logger.warning(f"aria2c download failed: {e}, falling back to urllib")

        # Fallback to urllib
        logger.info(f"Downloading Wikidata dump from {DUMP_URL}...")
        logger.info("TIP: Install aria2c for 10-20x faster downloads: brew install aria2")
        logger.info("This is a large file (~100GB) and will take significant time.")

        # Stream download with progress
        req = urllib.request.Request(
            DUMP_URL,
            headers={"User-Agent": "corp-extractor/1.0 (Wikidata dump importer)"}
        )

        with urllib.request.urlopen(req) as response:
            total = int(response.headers.get("content-length", 0))
            total_gb = total / (1024 ** 3) if total else 0

            with open(dump_path, "wb") as f:
                downloaded = 0
                chunk_size = 8 * 1024 * 1024  # 8MB chunks
                last_log_pct = 0

                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)

                    # Call progress callback if provided
                    if progress_callback:
                        progress_callback(downloaded, total)
                    else:
                        # Default logging (every 1%)
                        if total:
                            pct = int((downloaded / total) * 100)
                            if pct > last_log_pct:
                                downloaded_gb = downloaded / (1024 ** 3)
                                logger.info(f"Downloaded {downloaded_gb:.1f}GB / {total_gb:.1f}GB ({pct}%)")
                                last_log_pct = pct
                        elif downloaded % (1024 ** 3) < chunk_size:
                            # Log every GB if total unknown
                            downloaded_gb = downloaded / (1024 ** 3)
                            logger.info(f"Downloaded {downloaded_gb:.1f}GB")

        logger.info(f"Download complete: {dump_path}")
        self._dump_path = dump_path
        return dump_path

    def _download_with_aria2(
        self,
        output_path: Path,
        connections: int = 16,
    ) -> None:
        """
        Download using aria2c with multiple parallel connections.

        aria2c can achieve 10-20x faster downloads by using multiple
        connections to the server.

        Args:
            output_path: Where to save the downloaded file
            connections: Number of parallel connections (default: 16)
        """
        cmd = [
            "aria2c",
            "-x", str(connections),  # Max connections per server
            "-s", str(connections),  # Split file into N parts
            "-k", "10M",  # Min split size
            "--file-allocation=none",  # Faster on SSDs
            "-d", str(output_path.parent),
            "-o", output_path.name,
            "--console-log-level=notice",
            "--summary-interval=10",
            DUMP_URL,
        ]

        logger.info(f"Running: {' '.join(cmd)}")

        # Run aria2c and stream output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        # Stream output to logger
        if process.stdout:
            for line in process.stdout:
                line = line.strip()
                if line:
                    logger.info(f"aria2c: {line}")

        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f"aria2c exited with code {return_code}")

    def get_dump_path(self, target_dir: Optional[Path] = None) -> Path:
        """
        Get the path where the dump would be/is downloaded.

        Args:
            target_dir: Directory for the dump (default: ~/.cache/corp-extractor)

        Returns:
            Path to the dump file location
        """
        if target_dir is None:
            target_dir = Path.home() / ".cache" / "corp-extractor"
        return target_dir / "wikidata-latest-all.json.bz2"

    def iter_entities(self, dump_path: Optional[Path] = None) -> Iterator[dict]:
        """
        Stream entities from dump file, one at a time.

        Handles the Wikidata JSON dump format where each line after the opening
        bracket is a JSON object with a trailing comma (except the last).

        Args:
            dump_path: Path to dump file (uses self._dump_path if not provided)

        Yields:
            Parsed entity dictionaries
        """
        path = dump_path or self._dump_path
        if path is None:
            raise ValueError("No dump path provided. Call download_dump() first or pass dump_path.")

        path = Path(path)

        # Select opener based on extension
        if path.suffix == ".bz2":
            opener = bz2.open
        elif path.suffix == ".gz":
            opener = gzip.open
        else:
            # Assume uncompressed
            opener = open

        logger.info(f"Opening dump file: {path}")

        with opener(path, "rt", encoding="utf-8") as f:
            line_count = 0
            entity_count = 0

            for line in f:
                line_count += 1
                line = line.strip()

                # Skip array brackets
                if line in ("[", "]"):
                    continue

                # Strip trailing comma
                if line.endswith(","):
                    line = line[:-1]

                if not line:
                    continue

                try:
                    entity = json.loads(line)
                    entity_count += 1

                    if entity_count % 1_000_000 == 0:
                        logger.info(f"Processed {entity_count:,} entities...")

                    yield entity

                except json.JSONDecodeError as e:
                    logger.debug(f"Line {line_count}: JSON decode error: {e}")
                    continue

    def import_people(
        self,
        dump_path: Optional[Path] = None,
        limit: Optional[int] = None,
    ) -> Iterator[PersonRecord]:
        """
        Stream through dump, yielding ALL people with English Wikipedia articles.

        This method filters the dump for:
        - Items with type "item" (not properties)
        - Humans (P31 contains Q5)
        - Has English Wikipedia article (enwiki sitelink)

        PersonType is derived from positions (P39) and occupations (P106).

        Args:
            dump_path: Path to dump file (uses self._dump_path if not provided)
            limit: Optional maximum number of records to return

        Yields:
            PersonRecord for each qualifying person
        """
        path = dump_path or self._dump_path
        count = 0

        logger.info("Starting people import from Wikidata dump...")

        for entity in self.iter_entities(path):
            if limit and count >= limit:
                break

            record = self._process_person_entity(entity)
            if record:
                count += 1
                if count % 10_000 == 0:
                    logger.info(f"Yielded {count:,} people records...")
                yield record

        logger.info(f"People import complete: {count:,} records")

    def import_organizations(
        self,
        dump_path: Optional[Path] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CompanyRecord]:
        """
        Stream through dump, yielding organizations with English Wikipedia articles.

        This method filters the dump for:
        - Items with type "item"
        - Has P31 (instance of) matching an organization type
        - Has English Wikipedia article (enwiki sitelink)

        Args:
            dump_path: Path to dump file (uses self._dump_path if not provided)
            limit: Optional maximum number of records to return

        Yields:
            CompanyRecord for each qualifying organization
        """
        path = dump_path or self._dump_path
        count = 0

        logger.info("Starting organization import from Wikidata dump...")

        for entity in self.iter_entities(path):
            if limit and count >= limit:
                break

            record = self._process_org_entity(entity)
            if record:
                count += 1
                if count % 10_000 == 0:
                    logger.info(f"Yielded {count:,} organization records...")
                yield record

        logger.info(f"Organization import complete: {count:,} records")

    def _process_person_entity(self, entity: dict) -> Optional[PersonRecord]:
        """
        Process a single entity, return PersonRecord if it's a human with enwiki.

        Args:
            entity: Parsed Wikidata entity dictionary

        Returns:
            PersonRecord if entity qualifies, None otherwise
        """
        # Must be an item (not property)
        if entity.get("type") != "item":
            return None

        # Must be human (P31 contains Q5)
        if not self._is_human(entity):
            return None

        # Must have English Wikipedia article
        sitelinks = entity.get("sitelinks", {})
        if "enwiki" not in sitelinks:
            return None

        # Extract person data
        return self._extract_person_data(entity)

    def _process_org_entity(self, entity: dict) -> Optional[CompanyRecord]:
        """
        Process a single entity, return CompanyRecord if it's an organization with enwiki.

        Args:
            entity: Parsed Wikidata entity dictionary

        Returns:
            CompanyRecord if entity qualifies, None otherwise
        """
        # Must be an item (not property)
        if entity.get("type") != "item":
            return None

        # Get organization type from P31
        entity_type = self._get_org_type(entity)
        if entity_type is None:
            return None

        # Must have English Wikipedia article
        sitelinks = entity.get("sitelinks", {})
        if "enwiki" not in sitelinks:
            return None

        # Extract organization data
        return self._extract_org_data(entity, entity_type)

    def _is_human(self, entity: dict) -> bool:
        """
        Check if entity has P31 (instance of) = Q5 (human).

        Args:
            entity: Parsed Wikidata entity dictionary

        Returns:
            True if entity is a human
        """
        claims = entity.get("claims", {})
        for claim in claims.get("P31", []):
            mainsnak = claim.get("mainsnak", {})
            datavalue = mainsnak.get("datavalue", {})
            value = datavalue.get("value", {})
            if isinstance(value, dict) and value.get("id") == "Q5":
                return True
        return False

    def _get_org_type(self, entity: dict) -> Optional[EntityType]:
        """
        Check if entity has P31 (instance of) matching an organization type.

        Args:
            entity: Parsed Wikidata entity dictionary

        Returns:
            EntityType if entity is an organization, None otherwise
        """
        claims = entity.get("claims", {})
        for claim in claims.get("P31", []):
            mainsnak = claim.get("mainsnak", {})
            datavalue = mainsnak.get("datavalue", {})
            value = datavalue.get("value", {})
            if isinstance(value, dict):
                qid = value.get("id", "")
                if qid in ORG_TYPE_TO_ENTITY_TYPE:
                    return ORG_TYPE_TO_ENTITY_TYPE[qid]
        return None

    def _get_claim_values(self, entity: dict, prop: str) -> list[str]:
        """
        Get all QID values for a property (e.g., P39, P106).

        Args:
            entity: Parsed Wikidata entity dictionary
            prop: Property ID (e.g., "P39", "P106")

        Returns:
            List of QID strings
        """
        claims = entity.get("claims", {})
        values = []
        for claim in claims.get(prop, []):
            mainsnak = claim.get("mainsnak", {})
            datavalue = mainsnak.get("datavalue", {})
            value = datavalue.get("value", {})
            if isinstance(value, dict):
                qid = value.get("id")
                if qid:
                    values.append(qid)
        return values

    def _get_positions_with_org(self, claims: dict) -> list[dict]:
        """
        Extract P39 positions with qualifiers for org and dates.

        Qualifiers extracted:
        - P642 (of) - the organization
        - P580 (start time) - when the position started
        - P582 (end time) - when the position ended

        Args:
            claims: Claims dictionary from entity

        Returns:
            List of position dictionaries with position_qid, org_qid, start_date, end_date
        """
        positions = []
        for claim in claims.get("P39", []):
            mainsnak = claim.get("mainsnak", {})
            datavalue = mainsnak.get("datavalue", {})
            pos_value = datavalue.get("value", {})
            pos_qid = pos_value.get("id") if isinstance(pos_value, dict) else None
            if not pos_qid:
                continue

            qualifiers = claim.get("qualifiers", {})

            # Check for P642 qualifier (of - the organization)
            org_qid = None
            for qual in qualifiers.get("P642", []):
                qual_datavalue = qual.get("datavalue", {})
                qual_value = qual_datavalue.get("value", {})
                if isinstance(qual_value, dict):
                    org_qid = qual_value.get("id")
                    break

            # Check for P580 qualifier (start time)
            start_date = None
            for qual in qualifiers.get("P580", []):
                qual_datavalue = qual.get("datavalue", {})
                qual_value = qual_datavalue.get("value", {})
                if isinstance(qual_value, dict):
                    time_str = qual_value.get("time", "")
                    start_date = self._parse_time_value(time_str)
                    break

            # Check for P582 qualifier (end time)
            end_date = None
            for qual in qualifiers.get("P582", []):
                qual_datavalue = qual.get("datavalue", {})
                qual_value = qual_datavalue.get("value", {})
                if isinstance(qual_value, dict):
                    time_str = qual_value.get("time", "")
                    end_date = self._parse_time_value(time_str)
                    break

            positions.append({
                "position_qid": pos_qid,
                "org_qid": org_qid,
                "start_date": start_date,
                "end_date": end_date,
            })
        return positions

    def _parse_time_value(self, time_str: str) -> Optional[str]:
        """
        Parse Wikidata time value to ISO date string.

        Args:
            time_str: Wikidata time format like "+2020-01-15T00:00:00Z"

        Returns:
            ISO date string (YYYY-MM-DD) or None
        """
        if not time_str:
            return None
        # Remove leading + and extract date part
        time_str = time_str.lstrip("+")
        if "T" in time_str:
            return time_str.split("T")[0]
        return None

    def _classify_person_type(
        self,
        positions: list[dict],
        occupations: list[str],
    ) -> PersonType:
        """
        Determine PersonType from P39 positions and P106 occupations.

        Priority order:
        1. Check positions (more specific)
        2. Check occupations
        3. Default to UNKNOWN

        Args:
            positions: List of position dictionaries from _get_positions_with_org
            occupations: List of occupation QIDs from P106

        Returns:
            Classified PersonType
        """
        # Check positions first (more specific)
        for pos in positions:
            pos_qid = pos.get("position_qid", "")
            if pos_qid in EXECUTIVE_POSITION_QIDS:
                return PersonType.EXECUTIVE
            if pos_qid in POLITICIAN_POSITION_QIDS:
                return PersonType.POLITICIAN

        # Then check occupations
        for occ in occupations:
            if occ in OCCUPATION_TO_TYPE:
                return OCCUPATION_TO_TYPE[occ]

        # Default
        return PersonType.UNKNOWN

    def _get_best_role_org(
        self,
        positions: list[dict],
    ) -> tuple[str, str, str, Optional[str], Optional[str]]:
        """
        Select best position for role/org display.

        Priority:
        1. Positions with org and dates
        2. Positions with org
        3. Positions with dates
        4. Any position

        Args:
            positions: List of position dictionaries

        Returns:
            Tuple of (role_qid, org_label, org_qid, start_date, end_date)
            Note: In dump mode, we return QIDs since we don't have labels
        """
        # Priority 1: Position with org and dates
        for pos in positions:
            if pos.get("org_qid") and (pos.get("start_date") or pos.get("end_date")):
                return (
                    pos["position_qid"],
                    "",
                    pos["org_qid"],
                    pos.get("start_date"),
                    pos.get("end_date"),
                )

        # Priority 2: Position with org
        for pos in positions:
            if pos.get("org_qid"):
                return (
                    pos["position_qid"],
                    "",
                    pos["org_qid"],
                    pos.get("start_date"),
                    pos.get("end_date"),
                )

        # Priority 3: Position with dates
        for pos in positions:
            if pos.get("start_date") or pos.get("end_date"):
                return (
                    pos["position_qid"],
                    "",
                    pos.get("org_qid", ""),
                    pos.get("start_date"),
                    pos.get("end_date"),
                )

        # Priority 4: Any position
        if positions:
            pos = positions[0]
            return (
                pos["position_qid"],
                "",
                pos.get("org_qid", ""),
                pos.get("start_date"),
                pos.get("end_date"),
            )

        return "", "", "", None, None

    def _extract_person_data(self, entity: dict) -> Optional[PersonRecord]:
        """
        Extract PersonRecord from entity dict.

        Derives type/role/org from claims.

        Args:
            entity: Parsed Wikidata entity dictionary

        Returns:
            PersonRecord or None if essential data is missing
        """
        qid = entity.get("id", "")
        labels = entity.get("labels", {})
        label = labels.get("en", {}).get("value", "")

        if not label or not qid:
            return None

        claims = entity.get("claims", {})

        # Get positions (P39) with qualifiers for org
        positions = self._get_positions_with_org(claims)
        # Get occupations (P106)
        occupations = self._get_claim_values(entity, "P106")

        # Classify person type from positions + occupations
        person_type = self._classify_person_type(positions, occupations)

        # Get best role/org/dates from positions
        role_qid, _, org_qid, start_date, end_date = self._get_best_role_org(positions)

        # Get country (P27 - country of citizenship)
        countries = self._get_claim_values(entity, "P27")
        country_qid = countries[0] if countries else ""

        # Get description
        descriptions = entity.get("descriptions", {})
        description = descriptions.get("en", {}).get("value", "")

        # Track discovered organization
        if org_qid:
            self._discovered_orgs[org_qid] = ""  # Label not available from dump

        return PersonRecord(
            name=label,
            source="wikidata",
            source_id=qid,
            country=country_qid,  # Store QID, can resolve later
            person_type=person_type,
            known_for_role=role_qid,  # Store QID, can resolve later
            known_for_org="",  # Org label not available from dump
            from_date=start_date,
            to_date=end_date,
            record={
                "wikidata_id": qid,
                "label": label,
                "description": description,
                "positions": [p["position_qid"] for p in positions],
                "occupations": occupations,
                "org_qid": org_qid,
                "country_qid": country_qid,
            },
        )

    def _extract_org_data(
        self,
        entity: dict,
        entity_type: EntityType,
    ) -> Optional[CompanyRecord]:
        """
        Extract CompanyRecord from entity dict.

        Args:
            entity: Parsed Wikidata entity dictionary
            entity_type: Determined EntityType

        Returns:
            CompanyRecord or None if essential data is missing
        """
        qid = entity.get("id", "")
        labels = entity.get("labels", {})
        label = labels.get("en", {}).get("value", "")

        if not label or not qid:
            return None

        claims = entity.get("claims", {})

        # Get country (P17 - country)
        countries = self._get_claim_values(entity, "P17")
        country_qid = countries[0] if countries else ""

        # Get LEI (P1278)
        lei = self._get_string_claim(claims, "P1278")

        # Get ticker (P249)
        ticker = self._get_string_claim(claims, "P249")

        # Get description
        descriptions = entity.get("descriptions", {})
        description = descriptions.get("en", {}).get("value", "")

        # Get inception date (P571)
        inception = self._get_time_claim(claims, "P571")

        # Get dissolution date (P576)
        dissolution = self._get_time_claim(claims, "P576")

        return CompanyRecord(
            name=label,
            source="wikipedia",  # Use "wikipedia" per existing convention
            source_id=qid,
            region=country_qid,  # Store QID
            entity_type=entity_type,
            from_date=inception,
            to_date=dissolution,
            record={
                "wikidata_id": qid,
                "label": label,
                "description": description,
                "lei": lei,
                "ticker": ticker,
                "country_qid": country_qid,
            },
        )

    def _get_string_claim(self, claims: dict, prop: str) -> str:
        """
        Get first string value for a property.

        Args:
            claims: Claims dictionary
            prop: Property ID

        Returns:
            String value or empty string
        """
        for claim in claims.get(prop, []):
            mainsnak = claim.get("mainsnak", {})
            datavalue = mainsnak.get("datavalue", {})
            value = datavalue.get("value")
            if isinstance(value, str):
                return value
        return ""

    def _get_time_claim(self, claims: dict, prop: str) -> Optional[str]:
        """
        Get first time value for a property as ISO date string.

        Args:
            claims: Claims dictionary
            prop: Property ID

        Returns:
            ISO date string (YYYY-MM-DD) or None
        """
        for claim in claims.get(prop, []):
            mainsnak = claim.get("mainsnak", {})
            datavalue = mainsnak.get("datavalue", {})
            value = datavalue.get("value", {})
            if isinstance(value, dict):
                time_str = value.get("time", "")
                # Format: +2020-01-15T00:00:00Z
                if time_str:
                    # Remove leading + and extract date part
                    time_str = time_str.lstrip("+")
                    if "T" in time_str:
                        return time_str.split("T")[0]
        return None

    def get_discovered_organizations(self) -> list[CompanyRecord]:
        """
        Get organizations discovered during the people import.

        These are organizations associated with people (from P39 P642 qualifiers)
        that can be inserted into the organizations database if not already present.

        Note: In dump mode, we only have QIDs, not labels.

        Returns:
            List of CompanyRecord objects for discovered organizations
        """
        records = []
        for org_qid in self._discovered_orgs:
            record = CompanyRecord(
                name=org_qid,  # Only have QID, not label
                source="wikipedia",
                source_id=org_qid,
                region="",
                entity_type=EntityType.BUSINESS,  # Default
                record={
                    "wikidata_id": org_qid,
                    "discovered_from": "people_import",
                    "needs_label_resolution": True,
                },
            )
            records.append(record)
        logger.info(f"Discovered {len(records)} organizations from people import")
        return records

    def clear_discovered_organizations(self) -> None:
        """Clear the discovered organizations cache."""
        self._discovered_orgs.clear()

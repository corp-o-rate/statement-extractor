"""
Wikidata importer for the company database.

Imports company data from Wikidata using SPARQL queries
into the embedding database for company name matching.

Uses the public Wikidata Query Service endpoint.
"""

import json
import logging
import time
import urllib.parse
import urllib.request
from typing import Any, Iterator, Optional

from ..models import CompanyRecord

logger = logging.getLogger(__name__)

# Wikidata SPARQL endpoint
WIKIDATA_SPARQL_URL = "https://query.wikidata.org/sparql"

# SPARQL query to get companies with stock tickers or LEI codes
# This focuses on notable companies that are publicly traded or have LEI
COMPANY_SPARQL_QUERY = """
SELECT DISTINCT ?company ?companyLabel ?lei ?ticker ?exchange ?exchangeLabel ?country ?countryLabel ?inception WHERE {
  # Companies that are either:
  # - business enterprises (Q4830453)
  # - public companies (Q891723)
  # - or have a stock ticker
  {
    ?company wdt:P31/wdt:P279* wd:Q4830453.  # instance of business enterprise
    ?company wdt:P414 ?exchange.  # has stock exchange
  } UNION {
    ?company wdt:P31/wdt:P279* wd:Q891723.  # instance of public company
  } UNION {
    ?company wdt:P1278 ?lei.  # has LEI code
  }

  OPTIONAL { ?company wdt:P1278 ?lei. }  # LEI code
  OPTIONAL { ?company wdt:P249 ?ticker. }  # ticker symbol
  OPTIONAL { ?company wdt:P414 ?exchange. }  # stock exchange
  OPTIONAL { ?company wdt:P17 ?country. }  # country
  OPTIONAL { ?company wdt:P571 ?inception. }  # inception date

  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
LIMIT %d
OFFSET %d
"""

# Simpler query for bulk import - gets companies with English labels
BULK_COMPANY_QUERY = """
SELECT ?company ?companyLabel ?lei ?ticker ?country ?countryLabel WHERE {
  ?company wdt:P31/wdt:P279* wd:Q4830453.
  ?company rdfs:label ?companyLabel.
  FILTER(LANG(?companyLabel) = "en")

  OPTIONAL { ?company wdt:P1278 ?lei. }
  OPTIONAL { ?company wdt:P249 ?ticker. }
  OPTIONAL { ?company wdt:P17 ?country. }

  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
LIMIT %d
OFFSET %d
"""


class WikidataImporter:
    """
    Importer for Wikidata company data.

    Uses SPARQL queries against the public Wikidata Query Service
    to fetch companies with stock tickers, LEI codes, etc.
    """

    def __init__(self, batch_size: int = 5000, delay_seconds: float = 1.0):
        """
        Initialize the Wikidata importer.

        Args:
            batch_size: Number of records to fetch per SPARQL query
            delay_seconds: Delay between requests to be polite to the endpoint
        """
        self._batch_size = batch_size
        self._delay = delay_seconds

    def import_from_sparql(
        self,
        limit: Optional[int] = None,
        notable_only: bool = True,
    ) -> Iterator[CompanyRecord]:
        """
        Import company records from Wikidata via SPARQL.

        Args:
            limit: Optional limit on total records
            notable_only: If True, only fetch companies with tickers/LEI (default)

        Yields:
            CompanyRecord for each company
        """
        logger.info("Starting Wikidata company import via SPARQL...")

        query_template = COMPANY_SPARQL_QUERY if notable_only else BULK_COMPANY_QUERY

        offset = 0
        total_count = 0
        seen_ids = set()  # Track seen Wikidata IDs to avoid duplicates

        while True:
            if limit and total_count >= limit:
                break

            batch_limit = min(self._batch_size, (limit - total_count) if limit else self._batch_size)
            query = query_template % (batch_limit, offset)

            logger.info(f"Fetching Wikidata batch at offset {offset}...")

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

                record = self._parse_binding(binding)
                if record and record.source_id not in seen_ids:
                    seen_ids.add(record.source_id)
                    total_count += 1
                    batch_count += 1
                    yield record

            logger.info(f"Processed {batch_count} records from batch (total: {total_count})")

            if len(bindings) < batch_limit:
                # Last batch
                break

            offset += self._batch_size

            # Be polite to the endpoint
            if self._delay > 0:
                time.sleep(self._delay)

        logger.info(f"Completed Wikidata import: {total_count} records")

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
                "User-Agent": "corp-extractor/1.0 (company database builder)",
            }
        )

        with urllib.request.urlopen(req, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))

    def _parse_binding(self, binding: dict[str, Any]) -> Optional[CompanyRecord]:
        """Parse a SPARQL result binding into a CompanyRecord."""
        try:
            # Get Wikidata entity ID
            company_uri = binding.get("company", {}).get("value", "")
            if not company_uri:
                return None

            # Extract QID from URI (e.g., "http://www.wikidata.org/entity/Q312" -> "Q312")
            wikidata_id = company_uri.split("/")[-1]
            if not wikidata_id.startswith("Q"):
                return None

            # Get label
            label = binding.get("companyLabel", {}).get("value", "")
            if not label or label == wikidata_id:  # Skip if no English label
                return None

            # Get optional fields
            lei = binding.get("lei", {}).get("value")
            ticker = binding.get("ticker", {}).get("value")
            exchange_label = binding.get("exchangeLabel", {}).get("value")
            country_label = binding.get("countryLabel", {}).get("value")
            inception = binding.get("inception", {}).get("value")

            # Build record data
            record_data = {
                "wikidata_id": wikidata_id,
                "label": label,
            }
            if lei:
                record_data["lei"] = lei
            if ticker:
                record_data["ticker"] = ticker
            if exchange_label:
                record_data["exchange"] = exchange_label
            if country_label:
                record_data["country"] = country_label
            if inception:
                record_data["inception"] = inception

            return CompanyRecord(
                name=label.strip(),
                embedding_name=label.strip(),
                legal_name=label,
                source="wikipedia",  # Use "wikipedia" as source per schema
                source_id=wikidata_id,
                region=country_label or "",
                record=record_data,
            )

        except Exception as e:
            logger.debug(f"Failed to parse Wikidata binding: {e}")
            return None

    def search_company(self, name: str, limit: int = 10) -> list[CompanyRecord]:
        """
        Search for a specific company by name.

        Args:
            name: Company name to search for
            limit: Maximum results to return

        Returns:
            List of matching CompanyRecords
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

            # Check if it looks like a company
            company_keywords = ["company", "corporation", "inc", "ltd", "enterprise", "business"]
            if not any(kw in description.lower() for kw in company_keywords):
                continue

            record = CompanyRecord(
                name=label,
                embedding_name=label,
                legal_name=label,
                source="wikipedia",
                source_id=qid,
                region="",  # Not available from search API
                record={
                    "wikidata_id": qid,
                    "label": label,
                    "description": description,
                },
            )
            results.append(record)

        return results

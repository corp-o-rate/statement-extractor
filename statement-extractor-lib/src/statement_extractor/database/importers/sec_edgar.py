"""
SEC Edgar data importer for the company database.

Imports company data from SEC's company_tickers.json file
into the embedding database for company name matching.
"""

import json
import logging
from pathlib import Path
from typing import Any, Iterator, Optional

from ..models import CompanyRecord

logger = logging.getLogger(__name__)

# SEC Edgar company tickers URL
SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"


class SecEdgarImporter:
    """
    Importer for SEC Edgar company data.

    Uses the public company_tickers.json file from SEC.
    """

    def __init__(self):
        """Initialize the SEC Edgar importer."""
        pass

    def import_from_url(self, limit: Optional[int] = None) -> Iterator[CompanyRecord]:
        """
        Import records directly from SEC Edgar URL.

        Args:
            limit: Optional limit on number of records

        Yields:
            CompanyRecord for each company
        """
        import urllib.request

        logger.info(f"Downloading SEC Edgar data from {SEC_TICKERS_URL}")

        # SEC requires a User-Agent header
        req = urllib.request.Request(
            SEC_TICKERS_URL,
            headers={"User-Agent": "corp-extractor/1.0 (neil@corp-o-rate.com)"}
        )

        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode("utf-8"))

        yield from self._parse_tickers_data(data, limit)

    def import_from_file(
        self,
        file_path: str | Path,
        limit: Optional[int] = None,
    ) -> Iterator[CompanyRecord]:
        """
        Import records from a local SEC tickers JSON file.

        Args:
            file_path: Path to company_tickers.json
            limit: Optional limit on number of records

        Yields:
            CompanyRecord for each company
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"SEC Edgar file not found: {file_path}")

        logger.info(f"Importing SEC Edgar data from {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        yield from self._parse_tickers_data(data, limit)

    def _parse_tickers_data(
        self,
        data: dict[str, Any],
        limit: Optional[int],
    ) -> Iterator[CompanyRecord]:
        """
        Parse SEC tickers JSON data.

        Format: {"0": {"cik_str": "320193", "ticker": "AAPL", "title": "Apple Inc."}, ...}
        """
        count = 0

        for key, entry in data.items():
            if limit and count >= limit:
                break

            record = self._parse_entry(entry)
            if record:
                count += 1
                yield record

                if count % 1000 == 0:
                    logger.info(f"Imported {count} SEC Edgar records")

        logger.info(f"Completed SEC Edgar import: {count} records")

    def _parse_entry(self, entry: dict[str, Any]) -> Optional[CompanyRecord]:
        """Parse a single SEC ticker entry."""
        try:
            cik = entry.get("cik_str")
            ticker = entry.get("ticker", "")
            title = entry.get("title", "")

            if not cik or not title:
                return None

            # Normalize CIK to 10 digits with leading zeros
            cik_str = str(cik).zfill(10)

            # Use full title as name (preserve legal suffixes like Inc., Corp., etc.)
            name = title.strip()

            # Build record
            record_data = {
                "cik": cik_str,
                "ticker": ticker,
                "title": title,
            }

            return CompanyRecord(
                name=name,
                embedding_name=name,
                legal_name=title,
                source="sec_edgar",
                source_id=cik_str,
                record=record_data,
            )

        except Exception as e:
            logger.debug(f"Failed to parse SEC entry: {e}")
            return None

    def download_latest(self, output_path: Optional[Path] = None) -> Path:
        """
        Download the latest SEC tickers file.

        Args:
            output_path: Where to save the file

        Returns:
            Path to downloaded file
        """
        import tempfile
        import urllib.request

        if output_path is None:
            output_dir = Path(tempfile.gettempdir()) / "sec_edgar"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / "company_tickers.json"

        logger.info(f"Downloading SEC Edgar data from {SEC_TICKERS_URL}")

        # SEC requires a User-Agent header
        req = urllib.request.Request(
            SEC_TICKERS_URL,
            headers={"User-Agent": "corp-extractor/1.0 (contact@corp-o-rate.com)"}
        )
        with urllib.request.urlopen(req) as response:
            with open(output_path, "wb") as f:
                f.write(response.read())

        logger.info(f"Downloaded SEC Edgar data to {output_path}")
        return output_path

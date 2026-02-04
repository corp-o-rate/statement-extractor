"""
Tests for database canonicalization feature.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from statement_extractor.database.models import CompanyRecord, EntityType
from statement_extractor.database.store import (
    OrganizationDatabase,
    UnionFind,
    _normalize_for_canon,
    _expand_suffix,
    _names_match_for_canon,
    _normalize_region,
    _regions_match,
)


class TestCanonNormalization:
    """Tests for canonicalization name normalization functions."""

    def test_normalize_for_canon_lowercase(self):
        assert _normalize_for_canon("Apple Inc") == "apple inc"

    def test_normalize_for_canon_removes_trailing_dots(self):
        assert _normalize_for_canon("Amazon Ltd.") == "amazon ltd"

    def test_normalize_for_canon_normalizes_whitespace(self):
        assert _normalize_for_canon("Test  Corp  ") == "test corp"

    def test_expand_suffix_ltd(self):
        assert _expand_suffix("Amazon Ltd") == "amazon limited"

    def test_expand_suffix_corp(self):
        assert _expand_suffix("Test Corp") == "test corporation"

    def test_expand_suffix_inc(self):
        assert _expand_suffix("Apple Inc") == "apple incorporated"

    def test_expand_suffix_no_match(self):
        assert _expand_suffix("Google") == "google"

    def test_names_match_for_canon_exact(self):
        assert _names_match_for_canon("Apple Inc", "apple inc") is True

    def test_names_match_for_canon_suffix_expansion(self):
        assert _names_match_for_canon("Amazon Ltd", "Amazon Limited") is True
        assert _names_match_for_canon("Test Corp", "Test Corporation") is True

    def test_names_match_for_canon_different_names(self):
        assert _names_match_for_canon("Apple", "Microsoft") is False


class TestRegionNormalization:
    """Tests for region normalization using pycountry."""

    def test_normalize_region_alpha2(self):
        assert _normalize_region("GB") == "GB"
        assert _normalize_region("US") == "US"
        assert _normalize_region("DE") == "DE"

    def test_normalize_region_alpha3(self):
        assert _normalize_region("GBR") == "GB"
        assert _normalize_region("USA") == "US"
        assert _normalize_region("DEU") == "DE"

    def test_normalize_region_country_name(self):
        assert _normalize_region("United Kingdom") == "GB"
        assert _normalize_region("Germany") == "DE"
        assert _normalize_region("France") == "FR"

    def test_normalize_region_uk_aliases(self):
        """UK/England/etc should all normalize to GB."""
        assert _normalize_region("UK") == "GB"
        assert _normalize_region("U.K.") == "GB"
        assert _normalize_region("England") == "GB"
        assert _normalize_region("Great Britain") == "GB"

    def test_normalize_region_us_aliases(self):
        """USA aliases should normalize to US."""
        assert _normalize_region("USA") == "US"
        assert _normalize_region("U.S.A.") == "US"
        assert _normalize_region("U.S.") == "US"
        assert _normalize_region("America") == "US"

    def test_normalize_region_us_states(self):
        """US state codes/names should normalize to US (unless ambiguous with country)."""
        # Unambiguous state codes (not valid country codes) -> US
        assert _normalize_region("NY") == "US"
        assert _normalize_region("TX") == "US"
        assert _normalize_region("FL") == "US"

        # Ambiguous codes prefer country (CA=Canada, DE=Germany, etc.)
        assert _normalize_region("CA") == "CA"  # Canada, not California
        assert _normalize_region("DE") == "DE"  # Germany, not Delaware

        # Full state names always match US
        assert _normalize_region("California") == "US"
        assert _normalize_region("New York") == "US"
        assert _normalize_region("Texas") == "US"
        assert _normalize_region("Delaware") == "US"

    def test_normalize_region_empty(self):
        assert _normalize_region("") == ""
        assert _normalize_region("   ") == ""

    def test_normalize_region_unknown(self):
        """Unknown regions return empty string."""
        assert _normalize_region("XYZ123") == ""

    def test_regions_match_same(self):
        assert _regions_match("GB", "GB") is True
        assert _regions_match("UK", "GB") is True
        assert _regions_match("United Kingdom", "GB") is True

    def test_regions_match_different(self):
        assert _regions_match("GB", "US") is False
        assert _regions_match("UK", "Germany") is False

    def test_regions_match_empty_is_lenient(self):
        """Empty regions match anything (lenient for incomplete data)."""
        assert _regions_match("", "GB") is True
        assert _regions_match("GB", "") is True
        assert _regions_match("", "") is True


class TestUnionFind:
    """Tests for the UnionFind data structure."""

    def test_simple_union(self):
        uf = UnionFind([1, 2, 3, 4])
        uf.union(1, 2)
        assert uf.find(1) == uf.find(2)
        assert uf.find(3) != uf.find(1)

    def test_multiple_unions(self):
        uf = UnionFind([1, 2, 3, 4, 5])
        uf.union(1, 2)
        uf.union(3, 4)
        uf.union(2, 3)  # Should merge both groups

        # All should be in same group now
        root = uf.find(1)
        assert uf.find(2) == root
        assert uf.find(3) == root
        assert uf.find(4) == root
        assert uf.find(5) != root

    def test_groups(self):
        uf = UnionFind([1, 2, 3, 4, 5])
        uf.union(1, 2)
        uf.union(3, 4)

        groups = uf.groups()
        # Should have 3 groups: {1, 2}, {3, 4}, {5}
        group_sizes = sorted(len(g) for g in groups.values())
        assert group_sizes == [1, 2, 2]


class TestDatabaseCanonicalization:
    """Integration tests for database canonicalization."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            db = OrganizationDatabase(db_path=db_path, embedding_dim=4, readonly=False)
            yield db
            db.close()

    def _make_embedding(self) -> np.ndarray:
        """Create a small random embedding for testing."""
        return np.random.randn(4).astype(np.float32)

    def test_canonicalize_single_record(self, temp_db):
        """Single record should be canonical to itself."""
        record = CompanyRecord(
            name="Test Corp",
            source="sec_edgar",
            source_id="12345",
            entity_type=EntityType.BUSINESS,
            record={"ticker": "TST"},
        )
        temp_db.insert(record, self._make_embedding())

        result = temp_db.canonicalize()

        assert result["total_records"] == 1
        assert result["groups_found"] == 1
        assert result["multi_record_groups"] == 0
        assert result["records_updated"] == 1

    def test_canonicalize_by_name(self, temp_db):
        """Records with same normalized name should be grouped."""
        record1 = CompanyRecord(
            name="Amazon Inc",
            source="sec_edgar",
            source_id="amz-sec",
            entity_type=EntityType.BUSINESS,
            record={},
        )
        record2 = CompanyRecord(
            name="amazon inc",  # Same name, different case
            source="wikipedia",
            source_id="Q123",
            entity_type=EntityType.BUSINESS,
            record={},
        )

        temp_db.insert(record1, self._make_embedding())
        temp_db.insert(record2, self._make_embedding())

        result = temp_db.canonicalize()

        assert result["total_records"] == 2
        assert result["groups_found"] == 1  # Both in same group
        assert result["multi_record_groups"] == 1

        # Check canon stats
        stats = temp_db.get_canon_stats()
        assert stats["canonicalized_records"] == 2
        assert stats["multi_record_groups"] == 1

    def test_canonicalize_by_lei(self, temp_db):
        """Records with same LEI should be grouped."""
        record1 = CompanyRecord(
            name="Apple Inc",
            source="gleif",
            source_id="LEI12345",  # LEI is the source_id for gleif
            entity_type=EntityType.BUSINESS,
            record={},
        )
        record2 = CompanyRecord(
            name="APPLE CORPORATION",  # Different name
            source="sec_edgar",
            source_id="appl-sec",
            entity_type=EntityType.BUSINESS,
            record={"lei": "LEI12345"},  # Same LEI in record
        )

        temp_db.insert(record1, self._make_embedding())
        temp_db.insert(record2, self._make_embedding())

        result = temp_db.canonicalize()

        assert result["multi_record_groups"] == 1

    def test_canonicalize_by_ticker(self, temp_db):
        """Records with same ticker should be grouped."""
        record1 = CompanyRecord(
            name="Apple Inc",
            source="sec_edgar",
            source_id="appl-sec",
            entity_type=EntityType.BUSINESS,
            record={"ticker": "AAPL"},
        )
        record2 = CompanyRecord(
            name="Apple Corporation",
            source="wikipedia",
            source_id="Q123",
            entity_type=EntityType.BUSINESS,
            record={"ticker": "AAPL"},
        )

        temp_db.insert(record1, self._make_embedding())
        temp_db.insert(record2, self._make_embedding())

        result = temp_db.canonicalize()

        assert result["multi_record_groups"] == 1

    def test_canonicalize_source_priority(self, temp_db):
        """GLEIF should be canonical over SEC over Wikipedia."""
        record_wiki = CompanyRecord(
            name="Test Company",
            source="wikipedia",
            source_id="Q999",
            entity_type=EntityType.BUSINESS,
            record={},
        )
        record_sec = CompanyRecord(
            name="Test Company",
            source="sec_edgar",
            source_id="sec-123",
            entity_type=EntityType.BUSINESS,
            record={},
        )
        record_gleif = CompanyRecord(
            name="Test Company",
            source="gleif",
            source_id="LEI-999",
            entity_type=EntityType.BUSINESS,
            record={},
        )

        # Insert in reverse priority order
        temp_db.insert(record_wiki, self._make_embedding())
        temp_db.insert(record_sec, self._make_embedding())
        temp_db.insert(record_gleif, self._make_embedding())

        temp_db.canonicalize()

        # Verify GLEIF is the canonical record
        conn = temp_db._connect()
        cursor = conn.execute("""
            SELECT id, source, canon_id, canon_size FROM organizations
            WHERE source = 'gleif'
        """)
        gleif_row = cursor.fetchone()

        # GLEIF record should be canonical (canon_id = id) with size 3
        assert gleif_row["canon_id"] == gleif_row["id"]
        assert gleif_row["canon_size"] == 3

    def test_canonicalize_suffix_expansion(self, temp_db):
        """Records with equivalent suffix forms should be grouped."""
        record1 = CompanyRecord(
            name="Amazon Ltd",
            source="companies_house",
            source_id="UK123",
            entity_type=EntityType.BUSINESS,
            record={},
        )
        record2 = CompanyRecord(
            name="Amazon Limited",
            source="gleif",
            source_id="LEI-AMZ",
            entity_type=EntityType.BUSINESS,
            record={},
        )

        temp_db.insert(record1, self._make_embedding())
        temp_db.insert(record2, self._make_embedding())

        result = temp_db.canonicalize()

        assert result["multi_record_groups"] == 1

    def test_canonicalize_different_records_stay_separate(self, temp_db):
        """Records with different identifiers stay separate."""
        record1 = CompanyRecord(
            name="Apple Inc",
            source="sec_edgar",
            source_id="appl",
            entity_type=EntityType.BUSINESS,
            record={"ticker": "AAPL"},
        )
        record2 = CompanyRecord(
            name="Microsoft Corp",
            source="sec_edgar",
            source_id="msft",
            entity_type=EntityType.BUSINESS,
            record={"ticker": "MSFT"},
        )

        temp_db.insert(record1, self._make_embedding())
        temp_db.insert(record2, self._make_embedding())

        result = temp_db.canonicalize()

        assert result["total_records"] == 2
        assert result["groups_found"] == 2  # Each in own group
        assert result["multi_record_groups"] == 0

    def test_canonicalize_same_name_different_region_stay_separate(self, temp_db):
        """Same name in different regions should NOT be grouped."""
        record_uk = CompanyRecord(
            name="Test Company Ltd",
            source="companies_house",
            source_id="UK123",
            region="GB",
            entity_type=EntityType.BUSINESS,
            record={},
        )
        record_us = CompanyRecord(
            name="Test Company Ltd",
            source="sec_edgar",
            source_id="US456",
            region="US",
            entity_type=EntityType.BUSINESS,
            record={},
        )

        temp_db.insert(record_uk, self._make_embedding())
        temp_db.insert(record_us, self._make_embedding())

        result = temp_db.canonicalize()

        # Should be 2 separate groups - same name but different regions
        assert result["total_records"] == 2
        assert result["groups_found"] == 2
        assert result["multi_record_groups"] == 0

    def test_canonicalize_same_name_same_region_grouped(self, temp_db):
        """Same name in same region (after normalization) should be grouped."""
        record1 = CompanyRecord(
            name="Test Company",
            source="companies_house",
            source_id="UK123",
            region="GB",
            entity_type=EntityType.BUSINESS,
            record={},
        )
        record2 = CompanyRecord(
            name="test company",  # Same name different case
            source="wikipedia",
            source_id="Q999",
            region="UK",  # UK normalizes to GB
            entity_type=EntityType.BUSINESS,
            record={},
        )

        temp_db.insert(record1, self._make_embedding())
        temp_db.insert(record2, self._make_embedding())

        result = temp_db.canonicalize()

        # Should be 1 group - same name and same region after normalization
        assert result["total_records"] == 2
        assert result["groups_found"] == 1
        assert result["multi_record_groups"] == 1

    def test_canonicalize_us_states_normalized_to_us(self, temp_db):
        """Unambiguous US state codes should normalize to US for matching."""
        # Use NY (not a country code) instead of CA (which is Canada)
        record_ny = CompanyRecord(
            name="Acme Corp",
            source="sec_edgar",
            source_id="acme-ny",
            region="NY",  # New York (unambiguous US state)
            entity_type=EntityType.BUSINESS,
            record={},
        )
        record_us = CompanyRecord(
            name="Acme Corp",
            source="gleif",
            source_id="LEI-ACME",
            region="US",
            entity_type=EntityType.BUSINESS,
            record={},
        )

        temp_db.insert(record_ny, self._make_embedding())
        temp_db.insert(record_us, self._make_embedding())

        result = temp_db.canonicalize()

        # NY normalizes to US, so both should be in same group
        assert result["multi_record_groups"] == 1

    def test_canonicalize_lei_ignores_region(self, temp_db):
        """LEI matching should work regardless of region (LEI is globally unique)."""
        record1 = CompanyRecord(
            name="Global Corp",
            source="gleif",
            source_id="LEI-GLOBAL-123",
            region="DE",  # Germany
            entity_type=EntityType.BUSINESS,
            record={},
        )
        record2 = CompanyRecord(
            name="Global Corporation Inc",  # Different name
            source="sec_edgar",
            source_id="global-sec",
            region="US",  # Different region
            entity_type=EntityType.BUSINESS,
            record={"lei": "LEI-GLOBAL-123"},  # Same LEI
        )

        temp_db.insert(record1, self._make_embedding())
        temp_db.insert(record2, self._make_embedding())

        result = temp_db.canonicalize()

        # Should be grouped by LEI even though names and regions differ
        assert result["multi_record_groups"] == 1

    def test_get_record_by_id_includes_canon_fields(self, temp_db):
        """_get_record_by_id should include db_id and canon_id in record dict."""
        record = CompanyRecord(
            name="Test Corp",
            source="sec_edgar",
            source_id="12345",
            entity_type=EntityType.BUSINESS,
            record={},
        )
        row_id = temp_db.insert(record, self._make_embedding())
        temp_db.canonicalize()

        fetched = temp_db._get_record_by_id(row_id)

        assert fetched is not None
        assert "db_id" in fetched.record
        assert "canon_id" in fetched.record
        assert fetched.record["db_id"] == row_id
        assert fetched.record["canon_id"] == row_id  # Self-canonical

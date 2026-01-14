"""Tests for canonicalization module."""

import pytest

from statement_extractor.canonicalization import (
    Canonicalizer,
    default_entity_canonicalizer,
    deduplicate_statements_exact,
)
from statement_extractor.models import Entity, EntityType, Statement


class TestDefaultEntityCanonicalizer:
    """Tests for default_entity_canonicalizer function."""

    def test_trims_whitespace(self):
        assert default_entity_canonicalizer("  Apple Inc.  ") == "apple inc."

    def test_lowercases(self):
        assert default_entity_canonicalizer("APPLE INC.") == "apple inc."

    def test_removes_leading_the(self):
        assert default_entity_canonicalizer("The Apple Inc.") == "apple inc."

    def test_removes_leading_a(self):
        assert default_entity_canonicalizer("A new product") == "new product"

    def test_removes_leading_an(self):
        assert default_entity_canonicalizer("An iPhone") == "iphone"

    def test_normalizes_internal_whitespace(self):
        assert default_entity_canonicalizer("Apple   Inc.") == "apple inc."

    def test_handles_empty_string(self):
        assert default_entity_canonicalizer("") == ""

    def test_handles_only_determiner(self):
        assert default_entity_canonicalizer("the") == ""

    def test_preserves_non_determiner_start(self):
        assert default_entity_canonicalizer("Tesla Motors") == "tesla motors"

    def test_handles_mixed_case_determiner(self):
        assert default_entity_canonicalizer("THE Apple") == "apple"


class TestCanonicalizer:
    """Tests for Canonicalizer class."""

    def test_default_function(self):
        canon = Canonicalizer()
        assert canon.canonicalize_entity("The Apple Inc.") == "apple inc."

    def test_custom_function(self):
        canon = Canonicalizer(entity_fn=lambda x: x.upper())
        assert canon.canonicalize_entity("apple") == "APPLE"

    def test_canonicalize_statement_entities(self):
        canon = Canonicalizer()
        stmt = Statement(
            subject=Entity(text="The Apple Inc.", type=EntityType.ORG),
            predicate="announced",
            object=Entity(text="A new iPhone", type=EntityType.PRODUCT),
        )
        subj, obj = canon.canonicalize_statement_entities(stmt)
        assert subj == "apple inc."
        assert obj == "new iphone"

    def test_create_dedup_key(self):
        canon = Canonicalizer()
        stmt = Statement(
            subject=Entity(text="Apple Inc.", type=EntityType.ORG),
            predicate="Announced",
            object=Entity(text="iPhone 15", type=EntityType.PRODUCT),
        )
        key = canon.create_dedup_key(stmt)
        assert key == ("apple inc.", "announced", "iphone 15")

    def test_create_dedup_key_with_canonical_predicate(self):
        canon = Canonicalizer()
        stmt = Statement(
            subject=Entity(text="Apple Inc.", type=EntityType.ORG),
            predicate="bought",
            object=Entity(text="Beats", type=EntityType.ORG),
        )
        key = canon.create_dedup_key(stmt, predicate_canonical="acquired")
        assert key == ("apple inc.", "acquired", "beats")


class TestDeduplicateStatementsExact:
    """Tests for deduplicate_statements_exact function."""

    def _make_stmt(self, subj: str, pred: str, obj: str) -> Statement:
        return Statement(
            subject=Entity(text=subj, type=EntityType.ORG),
            predicate=pred,
            object=Entity(text=obj, type=EntityType.PRODUCT),
        )

    def test_removes_exact_duplicates(self):
        stmts = [
            self._make_stmt("Apple", "announced", "iPhone"),
            self._make_stmt("Apple", "announced", "iPhone"),
        ]
        result = deduplicate_statements_exact(stmts)
        assert len(result) == 1

    def test_keeps_first_occurrence(self):
        stmt1 = self._make_stmt("Apple", "announced", "iPhone")
        stmt1.source_text = "first"
        stmt2 = self._make_stmt("Apple", "announced", "iPhone")
        stmt2.source_text = "second"

        result = deduplicate_statements_exact([stmt1, stmt2])
        assert len(result) == 1
        assert result[0].source_text == "first"

    def test_case_insensitive_dedup(self):
        stmts = [
            self._make_stmt("Apple", "ANNOUNCED", "iPhone"),
            self._make_stmt("APPLE", "announced", "IPHONE"),
        ]
        result = deduplicate_statements_exact(stmts)
        assert len(result) == 1

    def test_keeps_different_predicates(self):
        stmts = [
            self._make_stmt("Apple", "announced", "iPhone"),
            self._make_stmt("Apple", "released", "iPhone"),
        ]
        result = deduplicate_statements_exact(stmts)
        assert len(result) == 2

    def test_keeps_different_objects(self):
        stmts = [
            self._make_stmt("Apple", "announced", "iPhone"),
            self._make_stmt("Apple", "announced", "iPad"),
        ]
        result = deduplicate_statements_exact(stmts)
        assert len(result) == 2

    def test_handles_empty_list(self):
        result = deduplicate_statements_exact([])
        assert result == []

    def test_handles_single_statement(self):
        stmt = self._make_stmt("Apple", "announced", "iPhone")
        result = deduplicate_statements_exact([stmt])
        assert len(result) == 1

    def test_custom_canonicalizer(self):
        # Custom canonicalizer that ignores everything after space
        def custom(text: str) -> str:
            return text.split()[0].lower() if text else ""

        stmts = [
            self._make_stmt("Apple Inc.", "announced", "iPhone 15"),
            self._make_stmt("Apple Corp", "announced", "iPhone Pro"),
        ]
        result = deduplicate_statements_exact(stmts, entity_canonicalizer=custom)
        # Both should match as "apple" -> "announced" -> "iphone"
        assert len(result) == 1

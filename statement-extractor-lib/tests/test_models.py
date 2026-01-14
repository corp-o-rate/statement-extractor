"""Tests for Pydantic models."""

import pytest
from pathlib import Path
import tempfile

from statement_extractor.models import (
    Entity,
    EntityType,
    ExtractionOptions,
    ExtractionResult,
    PredicateComparisonConfig,
    PredicateMatch,
    PredicateTaxonomy,
    ScoringConfig,
    Statement,
)


class TestEntityType:
    """Tests for EntityType enum."""

    def test_all_types_defined(self):
        expected_types = {
            "ORG", "PERSON", "GPE", "LOC", "PRODUCT", "EVENT",
            "WORK_OF_ART", "LAW", "DATE", "MONEY", "PERCENT",
            "QUANTITY", "UNKNOWN"
        }
        actual_types = {t.value for t in EntityType}
        assert actual_types == expected_types


class TestEntity:
    """Tests for Entity model."""

    def test_default_type_is_unknown(self):
        entity = Entity(text="test")
        assert entity.type == EntityType.UNKNOWN

    def test_str_representation(self):
        entity = Entity(text="Apple Inc.", type=EntityType.ORG)
        assert str(entity) == "Apple Inc. (ORG)"


class TestStatement:
    """Tests for Statement model."""

    def test_creates_statement(self):
        stmt = Statement(
            subject=Entity(text="Apple", type=EntityType.ORG),
            predicate="announced",
            object=Entity(text="iPhone", type=EntityType.PRODUCT),
        )
        assert stmt.subject.text == "Apple"
        assert stmt.predicate == "announced"
        assert stmt.object.text == "iPhone"

    def test_optional_fields_default_to_none(self):
        stmt = Statement(
            subject=Entity(text="Apple"),
            predicate="announced",
            object=Entity(text="iPhone"),
        )
        assert stmt.source_text is None
        assert stmt.confidence_score is None
        assert stmt.evidence_span is None
        assert stmt.canonical_predicate is None

    def test_str_representation(self):
        stmt = Statement(
            subject=Entity(text="Apple"),
            predicate="announced",
            object=Entity(text="iPhone"),
        )
        assert str(stmt) == "Apple -- announced --> iPhone"

    def test_as_triple(self):
        stmt = Statement(
            subject=Entity(text="Apple"),
            predicate="announced",
            object=Entity(text="iPhone"),
        )
        assert stmt.as_triple() == ("Apple", "announced", "iPhone")

    def test_confidence_score_validation(self):
        stmt = Statement(
            subject=Entity(text="Apple"),
            predicate="announced",
            object=Entity(text="iPhone"),
            confidence_score=0.5,
        )
        assert stmt.confidence_score == 0.5

        # Should raise for invalid values
        with pytest.raises(ValueError):
            Statement(
                subject=Entity(text="Apple"),
                predicate="announced",
                object=Entity(text="iPhone"),
                confidence_score=1.5,  # Invalid
            )


class TestExtractionResult:
    """Tests for ExtractionResult model."""

    def test_empty_result(self):
        result = ExtractionResult()
        assert len(result) == 0
        assert list(result) == []

    def test_with_statements(self):
        stmt = Statement(
            subject=Entity(text="Apple"),
            predicate="announced",
            object=Entity(text="iPhone"),
        )
        result = ExtractionResult(statements=[stmt, stmt])
        assert len(result) == 2

    def test_to_triples(self):
        stmt1 = Statement(
            subject=Entity(text="Apple"),
            predicate="announced",
            object=Entity(text="iPhone"),
        )
        stmt2 = Statement(
            subject=Entity(text="Google"),
            predicate="released",
            object=Entity(text="Pixel"),
        )
        result = ExtractionResult(statements=[stmt1, stmt2])

        triples = result.to_triples()
        assert triples == [
            ("Apple", "announced", "iPhone"),
            ("Google", "released", "Pixel"),
        ]


class TestPredicateTaxonomy:
    """Tests for PredicateTaxonomy model."""

    def test_creates_taxonomy(self):
        taxonomy = PredicateTaxonomy(predicates=["acquired", "founded"])
        assert len(taxonomy.predicates) == 2

    def test_from_list(self):
        taxonomy = PredicateTaxonomy.from_list(
            ["acquired", "founded"],
            name="business"
        )
        assert taxonomy.predicates == ["acquired", "founded"]
        assert taxonomy.name == "business"

    def test_from_file(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("acquired\n")
            f.write("founded\n")
            f.write("# comment line\n")
            f.write("works_for\n")
            f.write("\n")  # empty line
            f.flush()

            taxonomy = PredicateTaxonomy.from_file(f.name)

        assert taxonomy.predicates == ["acquired", "founded", "works_for"]

        # Cleanup
        Path(f.name).unlink()


class TestPredicateMatch:
    """Tests for PredicateMatch model."""

    def test_default_values(self):
        match = PredicateMatch(original="bought")
        assert match.original == "bought"
        assert match.canonical is None
        assert match.similarity == 0.0
        assert match.matched is False

    def test_matched_predicate(self):
        match = PredicateMatch(
            original="bought",
            canonical="acquired",
            similarity=0.85,
            matched=True,
        )
        assert match.matched
        assert match.canonical == "acquired"


class TestPredicateComparisonConfig:
    """Tests for PredicateComparisonConfig model."""

    def test_default_values(self):
        config = PredicateComparisonConfig()
        assert config.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
        assert config.similarity_threshold == 0.75
        assert config.dedup_threshold == 0.85
        assert config.normalize_text is True

    def test_custom_values(self):
        config = PredicateComparisonConfig(
            similarity_threshold=0.8,
            dedup_threshold=0.9,
        )
        assert config.similarity_threshold == 0.8
        assert config.dedup_threshold == 0.9


class TestScoringConfig:
    """Tests for ScoringConfig model."""

    def test_default_values(self):
        config = ScoringConfig()
        assert config.quality_weight == 1.0
        assert config.coverage_weight == 0.5
        assert config.redundancy_penalty == 0.3
        assert config.length_penalty == 0.1
        assert config.min_confidence == 0.0
        assert config.merge_top_n == 3

    def test_validation(self):
        # Valid
        config = ScoringConfig(min_confidence=0.5)
        assert config.min_confidence == 0.5

        # Invalid
        with pytest.raises(ValueError):
            ScoringConfig(min_confidence=1.5)


class TestExtractionOptions:
    """Tests for ExtractionOptions model."""

    def test_default_values(self):
        options = ExtractionOptions()
        assert options.num_beams == 4
        assert options.diversity_penalty == 1.0
        assert options.max_attempts == 3
        assert options.deduplicate is True
        assert options.merge_beams is True
        assert options.embedding_dedup is True

    def test_with_taxonomy(self):
        taxonomy = PredicateTaxonomy(predicates=["acquired"])
        options = ExtractionOptions(predicate_taxonomy=taxonomy)
        assert options.predicate_taxonomy is not None
        assert "acquired" in options.predicate_taxonomy.predicates

    def test_with_custom_canonicalizer(self):
        def custom(text: str) -> str:
            return text.upper()

        options = ExtractionOptions(entity_canonicalizer=custom)
        assert options.entity_canonicalizer is not None
        assert options.entity_canonicalizer("test") == "TEST"

    def test_disable_embeddings(self):
        options = ExtractionOptions(
            embedding_dedup=False,
            merge_beams=False,
        )
        assert not options.embedding_dedup
        assert not options.merge_beams

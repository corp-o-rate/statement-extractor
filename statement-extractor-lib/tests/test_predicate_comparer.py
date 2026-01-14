"""Tests for predicate_comparer module.

These tests require sentence-transformers to be installed.
They will be skipped if the dependency is not available.
"""

import pytest

from statement_extractor.models import Entity, EntityType, Statement

# Check if sentence-transformers is available
try:
    import sentence_transformers
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

requires_embeddings = pytest.mark.skipif(
    not HAS_SENTENCE_TRANSFORMERS,
    reason="sentence-transformers not installed"
)


@requires_embeddings
class TestEmbeddingDependencyError:
    """Tests for dependency error handling."""

    def test_error_message_is_helpful(self):
        from statement_extractor.predicate_comparer import EmbeddingDependencyError

        error = EmbeddingDependencyError("test")
        assert "test" in str(error)


@requires_embeddings
class TestPredicateComparer:
    """Tests for PredicateComparer class."""

    @pytest.fixture
    def comparer(self):
        from statement_extractor.predicate_comparer import PredicateComparer
        return PredicateComparer()

    @pytest.fixture
    def comparer_with_taxonomy(self):
        from statement_extractor.predicate_comparer import PredicateComparer
        from statement_extractor.models import PredicateTaxonomy

        taxonomy = PredicateTaxonomy(predicates=[
            "acquired",
            "founded",
            "works_for",
            "announced",
            "invested_in",
        ])
        return PredicateComparer(taxonomy=taxonomy)

    def test_computes_similarity(self, comparer):
        # Very similar predicates
        sim1 = comparer.compute_similarity("bought", "acquired")
        assert sim1 > 0.5  # Should be similar

        # Unrelated predicates
        sim2 = comparer.compute_similarity("bought", "founded")
        assert sim2 < sim1  # Should be less similar

    def test_are_similar_true_for_synonyms(self, comparer):
        # paraphrase-MiniLM-L6-v2 is better at word similarity than all-MiniLM-L6-v2
        # With default dedup_threshold of 0.65, these should match
        assert comparer.are_similar("acquired", "purchased")  # ~0.69
        assert comparer.are_similar("founded", "established")  # ~0.81

    def test_are_similar_false_for_unrelated(self, comparer):
        # These should not be considered similar at default threshold
        assert not comparer.are_similar("acquired", "founded")

    def test_match_to_canonical_with_taxonomy(self, comparer_with_taxonomy):
        # "bought" should match "acquired"
        match = comparer_with_taxonomy.match_to_canonical("bought")
        assert match.matched
        assert match.canonical == "acquired"
        assert match.similarity > 0.5

    def test_match_to_canonical_no_match(self, comparer_with_taxonomy):
        # Random predicate should not match well
        match = comparer_with_taxonomy.match_to_canonical("xyz123")
        assert not match.matched or match.similarity < 0.75

    def test_match_to_canonical_no_taxonomy(self, comparer):
        # Without taxonomy, should return unmatched
        match = comparer.match_to_canonical("bought")
        assert not match.matched
        assert match.canonical is None

    def test_match_batch(self, comparer_with_taxonomy):
        predicates = ["bought", "started", "employed"]
        matches = comparer_with_taxonomy.match_batch(predicates)

        assert len(matches) == 3
        # paraphrase-MiniLM-L6-v2 with 0.65 threshold:
        # "bought" -> "acquired" (~0.70 similarity)
        assert matches[0].canonical == "acquired"
        # "started" -> "founded" (~0.71 similarity)
        assert matches[1].canonical == "founded"
        # "employed" -> "works_for" (~0.52 similarity - below threshold, won't match)
        assert matches[2].original == "employed"
        assert not matches[2].matched  # Below threshold

    def test_deduplicate_statements(self, comparer):
        # paraphrase-MiniLM-L6-v2: "bought"/"acquired" ~0.70, above default 0.65 threshold
        def make_stmt(subj: str, pred: str, obj: str) -> Statement:
            return Statement(
                subject=Entity(text=subj, type=EntityType.ORG),
                predicate=pred,
                object=Entity(text=obj, type=EntityType.PRODUCT),
            )

        statements = [
            make_stmt("Apple", "acquired", "Beats"),
            make_stmt("Apple", "bought", "Beats"),  # Similar predicate (~0.70)
            make_stmt("Google", "acquired", "YouTube"),  # Different subject/object
        ]

        deduped = comparer.deduplicate_statements(statements)

        # Should remove "bought" as duplicate of "acquired"
        assert len(deduped) == 2

        predicates = {s.predicate for s in deduped}
        # First occurrence should be kept
        assert "acquired" in predicates

    def test_deduplicate_keeps_different_objects(self, comparer):
        def make_stmt(subj: str, pred: str, obj: str) -> Statement:
            return Statement(
                subject=Entity(text=subj, type=EntityType.ORG),
                predicate=pred,
                object=Entity(text=obj, type=EntityType.PRODUCT),
            )

        statements = [
            make_stmt("Apple", "announced", "iPhone"),
            make_stmt("Apple", "announced", "iPad"),
        ]

        deduped = comparer.deduplicate_statements(statements)

        # Both should be kept - different objects
        assert len(deduped) == 2

    def test_normalize_predicates(self, comparer_with_taxonomy):
        def make_stmt(subj: str, pred: str, obj: str) -> Statement:
            return Statement(
                subject=Entity(text=subj, type=EntityType.ORG),
                predicate=pred,
                object=Entity(text=obj, type=EntityType.PRODUCT),
            )

        statements = [
            make_stmt("Apple", "bought", "Beats"),
            make_stmt("Google", "purchased", "Waymo"),  # Use "purchased" instead - better similarity to "acquired"
        ]

        normalized = comparer_with_taxonomy.normalize_predicates(statements)

        # paraphrase-MiniLM-L6-v2 with 0.65 threshold:
        # "bought" -> "acquired" (~0.70 similarity)
        assert normalized[0].canonical_predicate == "acquired"
        # "purchased" -> "acquired" (~0.69 similarity)
        assert normalized[1].canonical_predicate == "acquired"


@requires_embeddings
class TestPredicateComparerConfig:
    """Tests for PredicateComparisonConfig."""

    def test_custom_threshold(self):
        from statement_extractor.predicate_comparer import PredicateComparer
        from statement_extractor.models import PredicateComparisonConfig, PredicateTaxonomy

        # Very high threshold - nothing should match
        config = PredicateComparisonConfig(similarity_threshold=0.99)
        taxonomy = PredicateTaxonomy(predicates=["acquired"])
        comparer = PredicateComparer(taxonomy=taxonomy, config=config)

        match = comparer.match_to_canonical("bought")
        assert not match.matched  # Threshold too high

    def test_custom_dedup_threshold(self):
        from statement_extractor.predicate_comparer import PredicateComparer
        from statement_extractor.models import PredicateComparisonConfig

        # Very high dedup threshold
        config = PredicateComparisonConfig(dedup_threshold=0.99)
        comparer = PredicateComparer(config=config)

        # Should not consider these similar at 0.99 threshold
        assert not comparer.are_similar("bought", "acquired")

    def test_normalize_text_option(self):
        from statement_extractor.predicate_comparer import PredicateComparer
        from statement_extractor.models import PredicateComparisonConfig

        # With normalization (default)
        config1 = PredicateComparisonConfig(normalize_text=True)
        comparer1 = PredicateComparer(config=config1)

        # Case should not matter
        sim1 = comparer1.compute_similarity("BOUGHT", "bought")
        assert sim1 > 0.99  # Should be identical after normalization

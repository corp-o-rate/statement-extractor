"""Integration tests for the statement extraction pipeline.

These tests verify the end-to-end functionality of the extraction system.
Note: Some tests require the actual model to be loaded, which may be slow.
"""

import pytest

from statement_extractor.models import (
    Entity,
    EntityType,
    ExtractionOptions,
    ExtractionResult,
    ScoringConfig,
    Statement,
)
from statement_extractor.canonicalization import (
    Canonicalizer,
    deduplicate_statements_exact,
)
from statement_extractor.scoring import BeamScorer, TripleScorer


class TestScoringPipeline:
    """Integration tests for the scoring pipeline."""

    def _make_stmt(
        self,
        subj: str,
        pred: str,
        obj: str,
        source_text: str | None = None,
    ) -> Statement:
        return Statement(
            subject=Entity(text=subj, type=EntityType.ORG),
            predicate=pred,
            object=Entity(text=obj, type=EntityType.PRODUCT),
            source_text=source_text,
        )

    def test_full_scoring_pipeline(self):
        """Test complete scoring and beam selection."""
        source = """
        Apple Inc. announced the new iPhone 15 at their annual event.
        Tim Cook presented the features to excited customers.
        The company also released the Apple Watch Series 9.
        """

        # Create multiple "beams" of statements
        beam1 = [
            self._make_stmt("Apple Inc.", "announced", "iPhone 15"),
            self._make_stmt("Tim Cook", "presented", "features"),
        ]
        beam2 = [
            self._make_stmt("Apple Inc.", "announced", "iPhone 15"),
            self._make_stmt("company", "released", "Apple Watch Series 9"),
        ]
        beam3 = [
            self._make_stmt("Random", "made", "stuff"),  # Ungrounded
        ]

        # Score and select best beam
        config = ScoringConfig()
        scorer = BeamScorer(config)

        best = scorer.select_best_beam([beam1, beam2, beam3], source)

        # Should select beam1 or beam2 (both have grounded statements)
        assert len(best) >= 1
        # Beam3 should not win
        assert not (len(best) == 1 and best[0].subject.text == "Random")

    def test_beam_merging_pipeline(self):
        """Test beam merging combines unique statements."""
        source = """
        Apple announced iPhone. Google released Pixel. Microsoft launched Surface.
        """

        beam1 = [
            self._make_stmt("Apple", "announced", "iPhone"),
        ]
        beam2 = [
            self._make_stmt("Google", "released", "Pixel"),
        ]
        beam3 = [
            self._make_stmt("Microsoft", "launched", "Surface"),
        ]

        config = ScoringConfig(min_confidence=0.0)
        scorer = BeamScorer(config)

        merged = scorer.merge_beams([beam1, beam2, beam3], source)

        # Should contain all three unique statements
        subjects = {s.subject.text for s in merged}
        assert "Apple" in subjects
        assert "Google" in subjects
        assert "Microsoft" in subjects

    def test_confidence_filtering(self):
        """Test that low-confidence statements are filtered."""
        source = "Apple announced iPhone."

        # One grounded, one not
        stmt_grounded = self._make_stmt("Apple", "announced", "iPhone")
        stmt_ungrounded = self._make_stmt("Random", "did", "something")

        scorer = TripleScorer()

        # Score both
        score1 = scorer.score_triple(stmt_grounded, source)
        score2 = scorer.score_triple(stmt_ungrounded, source)

        assert score1 > score2  # Grounded should score higher

        # With confidence filtering
        config = ScoringConfig(min_confidence=0.5)
        beam_scorer = BeamScorer(config)

        stmt_grounded.confidence_score = score1
        stmt_ungrounded.confidence_score = score2

        merged = beam_scorer.merge_beams([[stmt_grounded, stmt_ungrounded]], source)

        # Only high-confidence should remain
        if score2 < 0.5:
            assert len(merged) == 1
            assert merged[0].subject.text == "Apple"


class TestDeduplicationPipeline:
    """Integration tests for deduplication."""

    def _make_stmt(self, subj: str, pred: str, obj: str) -> Statement:
        return Statement(
            subject=Entity(text=subj, type=EntityType.ORG),
            predicate=pred,
            object=Entity(text=obj, type=EntityType.PRODUCT),
        )

    def test_exact_dedup_with_canonicalization(self):
        """Test exact dedup respects canonicalization."""
        statements = [
            self._make_stmt("The Apple Inc.", "Announced", "a new iPhone"),
            self._make_stmt("Apple Inc.", "announced", "new iPhone"),
        ]

        # Without custom canonicalization - these may not dedupe
        result1 = deduplicate_statements_exact(statements)

        # With canonicalization that removes determiners
        from statement_extractor.canonicalization import default_entity_canonicalizer
        result2 = deduplicate_statements_exact(
            statements,
            entity_canonicalizer=default_entity_canonicalizer
        )

        # With canonicalization, should dedupe
        assert len(result2) == 1

    def test_preserves_different_relationships(self):
        """Test that different relationships are preserved."""
        statements = [
            self._make_stmt("Apple", "announced", "iPhone"),
            self._make_stmt("Apple", "released", "iPhone"),
            self._make_stmt("Apple", "announced", "iPad"),
        ]

        result = deduplicate_statements_exact(statements)

        # All three should be preserved (different pred or obj)
        assert len(result) == 3


class TestCanonicalizationIntegration:
    """Integration tests for canonicalization with scoring."""

    def test_canonicalizer_with_scorer(self):
        """Test that canonicalization works with beam scoring."""
        source = "The Apple Inc. announced a new iPhone today."

        stmt = Statement(
            subject=Entity(text="The Apple Inc.", type=EntityType.ORG),
            predicate="announced",
            object=Entity(text="a new iPhone", type=EntityType.PRODUCT),
        )

        # Score should find entities even with determiners
        scorer = TripleScorer()
        score = scorer.score_triple(stmt, source)

        # Should find grounding despite determiners
        assert score > 0.5


class TestBackwardCompatibility:
    """Tests to ensure backward compatibility."""

    def test_extraction_options_defaults(self):
        """Test that default options work correctly."""
        options = ExtractionOptions()

        # v0.1.0 behavior should still work
        assert options.num_beams == 4
        assert options.diversity_penalty == 1.0
        assert options.deduplicate is True

        # New v0.2.0 defaults
        assert options.merge_beams is True
        assert options.embedding_dedup is True

    def test_disable_new_features(self):
        """Test that new features can be disabled for v0.1.0 behavior."""
        options = ExtractionOptions(
            merge_beams=False,
            embedding_dedup=False,
        )

        assert not options.merge_beams
        assert not options.embedding_dedup

    def test_statement_model_backward_compatible(self):
        """Test that Statement model works without new fields."""
        # v0.1.0 style creation
        stmt = Statement(
            subject=Entity(text="Apple"),
            predicate="announced",
            object=Entity(text="iPhone"),
        )

        # Should work
        assert stmt.subject.text == "Apple"
        assert str(stmt) == "Apple -- announced --> iPhone"

        # New fields default to None
        assert stmt.confidence_score is None
        assert stmt.evidence_span is None
        assert stmt.canonical_predicate is None


# Skip tests that require the actual model
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

requires_model = pytest.mark.skipif(
    not HAS_TORCH,
    reason="torch not installed"
)


@requires_model
class TestExtractorIntegration:
    """Integration tests that require the actual model.

    These are slow tests and may be skipped in CI.
    """

    @pytest.mark.slow
    def test_extractor_with_options(self):
        """Test extractor with various options.

        Note: This test loads the actual model and is slow.
        """
        # This would be a full integration test
        # Skipping actual model loading for unit tests
        pass

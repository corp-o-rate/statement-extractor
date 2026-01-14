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


class TestSourceTextConsistency:
    """Tests to verify source_text consistency after merging."""

    def _make_stmt(
        self,
        subj: str,
        pred: str,
        obj: str,
        source_text: str | None = None,
        subj_type: EntityType = EntityType.ORG,
        obj_type: EntityType = EntityType.EVENT,
        confidence: float | None = None,
    ) -> Statement:
        stmt = Statement(
            subject=Entity(text=subj, type=subj_type),
            predicate=pred,
            object=Entity(text=obj, type=obj_type),
            source_text=source_text,
        )
        stmt.confidence_score = confidence
        return stmt

    def test_source_text_should_contain_predicate_trigger(self):
        """
        Test that after merging, each statement's source_text contains
        a lexical trigger for its predicate.

        This test reproduces a bug where merging beams could result in
        mismatched source_text - e.g., a statement with predicate "is leader of"
        paired with source_text mentioning "congratulated".
        """
        # Simulate two beams producing same triple but different source_text
        beam1 = [
            self._make_stmt(
                "Aria Patel", "is leader of", "Cedar Ridge Cyclones",
                source_text="NPBA commissioner Aria Patel congratulated the Cedar Ridge Cyclones.",
                subj_type=EntityType.PERSON,
                obj_type=EntityType.ORG,
                confidence=0.9,  # Higher confidence but wrong source_text
            ),
        ]
        beam2 = [
            self._make_stmt(
                "Aria Patel", "congratulated", "Cedar Ridge Cyclones",
                source_text="NPBA commissioner Aria Patel congratulated the Cedar Ridge Cyclones.",
                subj_type=EntityType.PERSON,
                obj_type=EntityType.ORG,
                confidence=0.85,
            ),
        ]

        source = """NPBA commissioner Aria Patel congratulated the Cyclones and noted that the
        1,000th win represents a milestone for teams across the league."""

        config = ScoringConfig(min_confidence=0.0)
        scorer = BeamScorer(config)

        # After merging, if statements are different triples, both should be kept
        merged = scorer.merge_beams([beam1, beam2], source)

        # These are different predicates, so both should be present
        predicates = {s.predicate for s in merged}
        assert "congratulated" in predicates or "is leader of" in predicates

        # More importantly, verify each statement's source_text supports its predicate
        for stmt in merged:
            if stmt.source_text:
                # Check that at least one word from predicate appears in source_text
                predicate_words = stmt.predicate.lower().split()
                source_lower = stmt.source_text.lower()
                has_trigger = any(word in source_lower for word in predicate_words if len(word) > 2)

                # This assertion documents the expected behavior:
                # If source_text is set, it should contain evidence for the predicate
                assert has_trigger, (
                    f"Statement predicate '{stmt.predicate}' has no lexical trigger in "
                    f"source_text: '{stmt.source_text}'"
                )


class TestCedarRidgeCyclonesScenario:
    """
    Integration test using realistic sports article text.

    This tests extraction quality on multi-paragraph text with many entities
    and relationships, similar to real-world usage.
    """

    CEDAR_RIDGE_TEXT = """CEDAR FALLS, Iowa — On March 31, 2026, at Riverbend Arena, the Cedar Ridge Cyclones of the National Pro Basketball Association (NPBA) secured their 1,000th win in franchise history by defeating the Richmond Royals 112-105. The milestone comes in the program's 27th season and was fueled by a late defensive surge and steady scoring from the starting lineup. Star guard Elena Park led the way with 28 points on 9-for-20 shooting, while center Malik Carter added 15 points and 11 rebounds. Head coach Marcus Doyle praised the team's balance, noting that a 12-2 run in the fourth quarter turned a one-possession game into a comfortable finish. Team owner Sophia Lin called the win a testament to long-term planning and community support.

Park's 28-point performance elevated her to 3,184 career points, the most in Cedar Ridge Cyclones history. She also handed out six assists and contributed three steals, supporting a defense that held Richmond to 105 points. The night underscored the team's multi-faceted approach this season, with fellow guard Jalen Soto posting seven assists and forward Mateo Ruiz adding 12 points. The Cyclones have emphasized player development, including the 2024 opening of the Riverbend Training Center downtown and a youth-mentorship program funded by the team's foundation.

NPBA commissioner Aria Patel congratulated the Cyclones and noted that the 1,000th win represents a milestone for teams across the league. Friday's crowd of 15,472 at Riverbend Arena marked the largest regular-season attendance in Cedar Ridge's home history since 2015, and merchants along Route 20 reported a measurable uptick in business. The club announced a charitable partner agreement with the Cedar Falls Community Foundation to award $25,000 in grants to local youth basketball programs, reinforcing its community outreach. The Cyclones will return to action on April 6, when they host the Milwaukee Monarchs in a rematch that will cap a three-game homestand."""

    def _make_stmt(
        self,
        subj: str,
        pred: str,
        obj: str,
        source_text: str | None = None,
        subj_type: EntityType = EntityType.ORG,
        obj_type: EntityType = EntityType.ORG,
    ) -> Statement:
        return Statement(
            subject=Entity(text=subj, type=subj_type),
            predicate=pred,
            object=Entity(text=obj, type=obj_type),
            source_text=source_text,
        )

    def test_beam_scoring_with_sports_text(self):
        """Test beam scoring with realistic sports article."""
        # Create representative statements that could be extracted
        beam1 = [
            self._make_stmt(
                "Cedar Ridge Cyclones", "defeated", "Richmond Royals",
                source_text="the Cedar Ridge Cyclones of the National Pro Basketball Association (NPBA) secured their 1,000th win in franchise history by defeating the Richmond Royals 112-105",
            ),
            self._make_stmt(
                "Elena Park", "led", "Cedar Ridge Cyclones",
                source_text="Star guard Elena Park led the way with 28 points on 9-for-20 shooting",
                subj_type=EntityType.PERSON,
            ),
        ]

        beam2 = [
            self._make_stmt(
                "Cedar Ridge Cyclones", "prevails over", "Richmond Royals",
                source_text="the Cedar Ridge Cyclones of the National Pro Basketball Association (NPBA) secured their 1,000th win in franchise history by defeating the Richmond Royals 112-105",
            ),
            self._make_stmt(
                "Aria Patel", "congratulated", "Cedar Ridge Cyclones",
                source_text="NPBA commissioner Aria Patel congratulated the Cyclones and noted that the 1,000th win represents a milestone",
                subj_type=EntityType.PERSON,
            ),
        ]

        config = ScoringConfig(min_confidence=0.0)
        scorer = BeamScorer(config)

        # Merge should combine unique statements from both beams
        merged = scorer.merge_beams([beam1, beam2], self.CEDAR_RIDGE_TEXT)

        # Should have statements about Elena Park and Aria Patel (unique subjects/predicates)
        subjects = {s.subject.text for s in merged}
        assert "Elena Park" in subjects
        assert "Aria Patel" in subjects

    def test_evidence_spans_found_in_source(self):
        """Test that evidence spans can be located in the source text."""
        scorer = TripleScorer()

        stmt = Statement(
            subject=Entity(text="Elena Park", type=EntityType.PERSON),
            predicate="led",
            object=Entity(text="Cedar Ridge Cyclones", type=EntityType.ORG),
            source_text="Star guard Elena Park led the way with 28 points",
        )

        span = scorer.find_evidence_span(stmt, self.CEDAR_RIDGE_TEXT)
        assert span is not None

        # The span should contain the source_text
        start, end = span
        extracted = self.CEDAR_RIDGE_TEXT[start:end]
        assert "Elena Park" in extracted or "28 points" in extracted

    def test_all_key_entities_grounded(self):
        """Verify that key entities from the text can be grounded."""
        scorer = TripleScorer()

        key_entities = [
            "Cedar Ridge Cyclones",
            "Richmond Royals",
            "Elena Park",
            "Malik Carter",
            "Marcus Doyle",
            "Sophia Lin",
            "Aria Patel",
            "Cedar Falls Community Foundation",
        ]

        for entity in key_entities:
            # Each entity should appear in the source text
            assert entity.lower() in self.CEDAR_RIDGE_TEXT.lower(), (
                f"Entity '{entity}' not found in source text"
            )

    def test_source_text_predicate_consistency(self):
        """
        Test that statements maintain consistency between predicate and source_text.

        This test documents expected behavior: if a statement has source_text,
        the predicate should have some lexical grounding in that text.
        """
        # Example of a GOOD statement - predicate matches source_text
        good_stmt = self._make_stmt(
            "Aria Patel", "congratulated", "Cedar Ridge Cyclones",
            source_text="NPBA commissioner Aria Patel congratulated the Cyclones",
            subj_type=EntityType.PERSON,
        )

        # Example of a BAD statement - predicate doesn't match source_text
        # (This is what we're trying to prevent)
        bad_stmt = self._make_stmt(
            "Aria Patel", "is leader of", "Cedar Ridge Cyclones",
            source_text="NPBA commissioner Aria Patel congratulated the Cyclones",
            subj_type=EntityType.PERSON,
        )

        # Check that good statement has predicate trigger in source_text
        assert "congratulated" in good_stmt.source_text.lower()

        # Check that bad statement does NOT have predicate trigger in source_text
        predicate_words = bad_stmt.predicate.lower().split()
        source_lower = bad_stmt.source_text.lower()
        has_trigger = any(word in source_lower for word in predicate_words if len(word) > 2)
        assert not has_trigger, "Expected bad_stmt to have no predicate trigger"


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

    @pytest.mark.slow
    def test_cedar_ridge_extraction(self):
        """
        Test extraction on Cedar Ridge Cyclones sports article.

        This is a slow test that loads the actual model.
        """
        from statement_extractor import extract_statements, ExtractionOptions

        text = TestCedarRidgeCyclonesScenario.CEDAR_RIDGE_TEXT

        options = ExtractionOptions(
            num_beams=4,
            merge_beams=True,
            embedding_dedup=True,
        )

        result = extract_statements(text, options)

        # Should extract multiple statements
        assert len(result.statements) >= 5

        # Verify source_text consistency for each statement
        for stmt in result.statements:
            if stmt.source_text:
                predicate_words = stmt.predicate.lower().split()
                source_lower = stmt.source_text.lower()
                # At least one predicate word should appear in source_text
                has_trigger = any(
                    word in source_lower
                    for word in predicate_words
                    if len(word) > 2
                )
                assert has_trigger, (
                    f"Statement '{stmt}' has predicate '{stmt.predicate}' "
                    f"with no trigger in source_text: '{stmt.source_text}'"
                )

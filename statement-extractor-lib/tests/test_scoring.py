"""Tests for scoring module."""

import pytest

from statement_extractor.models import Entity, EntityType, ScoringConfig, Statement
from statement_extractor.scoring import BeamScorer, TripleScorer


class TestTripleScorer:
    """Tests for TripleScorer class."""

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

    def test_scores_grounded_triple_high(self):
        scorer = TripleScorer()
        source = "Apple Inc. announced the new iPhone 15 at the event."
        stmt = self._make_stmt("Apple Inc.", "announced", "iPhone 15")

        score = scorer.score_triple(stmt, source)
        assert score > 0.7  # Should be high since all parts appear in source

    def test_scores_ungrounded_triple_low(self):
        scorer = TripleScorer()
        source = "The weather is nice today."
        stmt = self._make_stmt("Apple Inc.", "announced", "iPhone 15")

        score = scorer.score_triple(stmt, source)
        assert score < 0.3  # Should be low since nothing appears in source

    def test_partial_grounding(self):
        scorer = TripleScorer()
        source = "Apple Inc. released something new."
        stmt = self._make_stmt("Apple Inc.", "announced", "iPhone 15")

        score = scorer.score_triple(stmt, source)
        # Subject found, predicate and object not
        assert 0.2 < score < 0.6

    def test_handles_empty_source(self):
        scorer = TripleScorer()
        stmt = self._make_stmt("Apple", "announced", "iPhone")

        score = scorer.score_triple(stmt, "")
        assert score == 0.5  # Neutral score for empty source

    def test_finds_evidence_span_with_source_text(self):
        scorer = TripleScorer()
        source = "First sentence. Apple announced iPhone. Last sentence."
        stmt = self._make_stmt("Apple", "announced", "iPhone")
        stmt.source_text = "Apple announced iPhone"

        span = scorer.find_evidence_span(stmt, source)
        assert span is not None
        start, end = span
        assert "Apple announced iPhone" in source[start:end]

    def test_finds_evidence_span_without_source_text(self):
        scorer = TripleScorer()
        source = "First sentence. Apple announced iPhone. Last sentence."
        stmt = self._make_stmt("Apple", "announced", "iPhone")

        span = scorer.find_evidence_span(stmt, source)
        assert span is not None
        start, end = span
        assert "Apple" in source[start:end]
        assert "iPhone" in source[start:end]

    def test_returns_none_for_ungrounded(self):
        scorer = TripleScorer()
        source = "The weather is nice."
        stmt = self._make_stmt("Apple", "announced", "iPhone")

        span = scorer.find_evidence_span(stmt, source)
        assert span is None

    def test_proximity_same_sentence(self):
        scorer = TripleScorer()
        source = "Apple announced iPhone today."
        stmt = self._make_stmt("Apple", "announced", "iPhone")

        score = scorer.score_triple(stmt, source)
        # High score because all in same sentence
        assert score > 0.7

    def test_proximity_different_sentences(self):
        scorer = TripleScorer()
        source = "Apple is a company. They announced something. The iPhone was released."
        stmt = self._make_stmt("Apple", "announced", "iPhone")

        score = scorer.score_triple(stmt, source)
        # Lower score because entities are spread across sentences
        assert score < 0.8


class TestBeamScorer:
    """Tests for BeamScorer class."""

    def _make_stmt(
        self,
        subj: str,
        pred: str,
        obj: str,
        confidence: float | None = None,
    ) -> Statement:
        stmt = Statement(
            subject=Entity(text=subj, type=EntityType.ORG),
            predicate=pred,
            object=Entity(text=obj, type=EntityType.PRODUCT),
        )
        stmt.confidence_score = confidence
        return stmt

    def test_scores_beam_with_quality(self):
        config = ScoringConfig(quality_weight=1.0, coverage_weight=0.0, redundancy_penalty=0.0)
        scorer = BeamScorer(config)

        source = "Apple announced iPhone. Google released Pixel."
        statements = [
            self._make_stmt("Apple", "announced", "iPhone", confidence=0.9),
            self._make_stmt("Google", "released", "Pixel", confidence=0.8),
        ]

        score = scorer.score_beam(statements, source)
        assert score > 0  # Should be positive with good confidence

    def test_computes_redundancy(self):
        scorer = BeamScorer()

        # Same subject and predicate - should be redundant
        statements = [
            self._make_stmt("Apple", "announced", "iPhone"),
            self._make_stmt("Apple", "announced", "iPad"),
        ]

        redundancy = scorer.compute_redundancy(statements)
        assert redundancy == 0  # Different objects, not redundant

        # Exact same - should be redundant
        statements = [
            self._make_stmt("Apple", "announced", "iPhone"),
            self._make_stmt("Apple", "announced", "iPhone"),
        ]

        redundancy = scorer.compute_redundancy(statements)
        assert redundancy > 0

    def test_select_best_beam(self):
        config = ScoringConfig()
        scorer = BeamScorer(config)

        source = "Apple announced iPhone today."

        # Beam 1: Well-grounded statement
        beam1 = [self._make_stmt("Apple", "announced", "iPhone")]

        # Beam 2: Poorly grounded statement
        beam2 = [self._make_stmt("Google", "released", "Pixel")]

        best = scorer.select_best_beam([beam1, beam2], source)
        # Beam 1 should win because it's grounded in source
        assert len(best) == 1
        assert best[0].subject.text == "Apple"

    def test_merge_beams_pools_statements(self):
        config = ScoringConfig(min_confidence=0.0)
        scorer = BeamScorer(config)

        source = "Apple announced iPhone. Google released Pixel."

        beam1 = [self._make_stmt("Apple", "announced", "iPhone")]
        beam2 = [self._make_stmt("Google", "released", "Pixel")]

        merged = scorer.merge_beams([beam1, beam2], source)
        # Should include statements from both beams
        assert len(merged) == 2

    def test_merge_beams_deduplicates(self):
        config = ScoringConfig(min_confidence=0.0)
        scorer = BeamScorer(config)

        source = "Apple announced iPhone."

        beam1 = [self._make_stmt("Apple", "announced", "iPhone")]
        beam2 = [self._make_stmt("Apple", "announced", "iPhone")]

        merged = scorer.merge_beams([beam1, beam2], source)
        # Duplicates should be removed
        assert len(merged) == 1

    def test_merge_beams_keeps_multiple_objects(self):
        """Test that same subject+predicate with different objects are kept."""
        config = ScoringConfig(min_confidence=0.0)
        scorer = BeamScorer(config)

        source = "Apple announced iPhone and iPad."

        beam1 = [self._make_stmt("Apple", "announced", "iPhone")]
        beam2 = [self._make_stmt("Apple", "announced", "iPad")]

        merged = scorer.merge_beams([beam1, beam2], source)
        # Both should be kept - different objects are valid
        assert len(merged) == 2

    def test_merge_beams_filters_by_confidence(self):
        config = ScoringConfig(min_confidence=0.5)
        scorer = BeamScorer(config)

        source = "Apple announced iPhone."

        stmt_high = self._make_stmt("Apple", "announced", "iPhone")
        stmt_high.confidence_score = 0.8

        stmt_low = self._make_stmt("Google", "released", "Pixel")
        stmt_low.confidence_score = 0.2

        merged = scorer.merge_beams([[stmt_high], [stmt_low]], source)
        # Only high-confidence statement should remain
        assert len(merged) == 1
        assert merged[0].subject.text == "Apple"

    def test_handles_empty_candidates(self):
        scorer = BeamScorer()
        source = "Some text."

        assert scorer.select_best_beam([], source) == []
        assert scorer.merge_beams([], source) == []

    def test_score_and_rank_statements(self):
        scorer = BeamScorer()
        source = "Apple announced iPhone. Weather is nice."

        stmt1 = self._make_stmt("Apple", "announced", "iPhone")
        stmt2 = self._make_stmt("Weather", "is", "nice")

        ranked = scorer.score_and_rank_statements([stmt1, stmt2], source)

        # All should have confidence scores now
        assert all(s.confidence_score is not None for s in ranked)
        # Should be sorted by confidence (descending)
        assert ranked[0].confidence_score >= ranked[1].confidence_score

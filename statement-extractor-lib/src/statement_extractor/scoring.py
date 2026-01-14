"""
Scoring module for statement extraction quality assessment.

Provides:
- TripleScorer: Score individual triples for groundedness
- BeamScorer: Score and select/merge beams based on quality metrics
"""

from typing import Optional

from .models import ScoringConfig, Statement


class TripleScorer:
    """
    Score individual triples for groundedness in source text.

    Groundedness is measured by checking:
    - Subject text appears in source
    - Object text appears in source
    - Subject and object are in proximity (same/nearby sentences)
    - Evidence span exists and is valid
    """

    def __init__(self, config: Optional[ScoringConfig] = None):
        self.config = config or ScoringConfig()

    def score_triple(self, statement: Statement, source_text: str) -> float:
        """
        Score a triple's groundedness (0-1).

        Higher scores indicate better grounding in source text.
        """
        if not source_text:
            return 0.5  # Neutral score if no source text

        score = 0.0
        weights_sum = 0.0

        # Check subject appears in source (weight: 0.3)
        subject_found = self._text_appears_in(statement.subject.text, source_text)
        score += 0.3 * (1.0 if subject_found else 0.0)
        weights_sum += 0.3

        # Check object appears in source (weight: 0.3)
        object_found = self._text_appears_in(statement.object.text, source_text)
        score += 0.3 * (1.0 if object_found else 0.0)
        weights_sum += 0.3

        # Check predicate has lexical trigger (weight: 0.2)
        predicate_grounded = self._predicate_has_trigger(statement.predicate, source_text)
        score += 0.2 * (1.0 if predicate_grounded else 0.0)
        weights_sum += 0.2

        # Check proximity - subject and object in same/nearby region (weight: 0.2)
        if subject_found and object_found:
            proximity_score = self._compute_proximity(
                statement.subject.text,
                statement.object.text,
                source_text
            )
            score += 0.2 * proximity_score
        weights_sum += 0.2

        return score / weights_sum if weights_sum > 0 else 0.0

    def find_evidence_span(
        self,
        statement: Statement,
        source_text: str
    ) -> Optional[tuple[int, int]]:
        """
        Find character offsets where the triple is grounded in source text.

        Returns (start, end) tuple or None if not found.
        """
        if not source_text:
            return None

        # If statement has source_text field, try to find it
        if statement.source_text:
            pos = source_text.lower().find(statement.source_text.lower())
            if pos >= 0:
                return (pos, pos + len(statement.source_text))

        # Otherwise, find the region containing both subject and object
        subject_lower = statement.subject.text.lower()
        object_lower = statement.object.text.lower()
        source_lower = source_text.lower()

        subj_pos = source_lower.find(subject_lower)
        obj_pos = source_lower.find(object_lower)

        if subj_pos >= 0 and obj_pos >= 0:
            start = min(subj_pos, obj_pos)
            end = max(
                subj_pos + len(subject_lower),
                obj_pos + len(object_lower)
            )
            # Extend to sentence boundaries
            start, end = self._extend_to_sentence(source_text, start, end)
            return (start, end)

        return None

    def _text_appears_in(self, text: str, source: str) -> bool:
        """Check if text appears in source (case-insensitive)."""
        return text.lower() in source.lower()

    def _predicate_has_trigger(self, predicate: str, source: str) -> bool:
        """Check if predicate has a lexical trigger in source."""
        # Extract main verb/word from predicate
        words = predicate.lower().split()
        source_lower = source.lower()

        # Check if any predicate word appears in source
        for word in words:
            if len(word) > 2 and word in source_lower:
                return True
        return False

    def _compute_proximity(
        self,
        subject_text: str,
        object_text: str,
        source: str
    ) -> float:
        """
        Compute proximity score (0-1) based on distance between subject and object.

        Returns 1.0 if same sentence, decreasing with distance.
        """
        source_lower = source.lower()
        subj_pos = source_lower.find(subject_text.lower())
        obj_pos = source_lower.find(object_text.lower())

        if subj_pos < 0 or obj_pos < 0:
            return 0.0

        # Check if in same sentence
        start = min(subj_pos, obj_pos)
        end = max(subj_pos, obj_pos)
        region = source[start:end]

        # If no sentence boundary between them, high proximity
        if '.' not in region and '!' not in region and '?' not in region:
            return 1.0

        # Otherwise, score decreases with distance
        # Assume ~100 chars per sentence on average
        sentence_distance = region.count('.') + region.count('!') + region.count('?')
        return max(0.0, 1.0 - (sentence_distance * 0.2))

    def _extend_to_sentence(
        self,
        source: str,
        start: int,
        end: int
    ) -> tuple[int, int]:
        """Extend span to sentence boundaries."""
        # Find sentence start
        sentence_start = start
        while sentence_start > 0:
            char = source[sentence_start - 1]
            if char in '.!?\n':
                break
            sentence_start -= 1

        # Find sentence end
        sentence_end = end
        while sentence_end < len(source):
            char = source[sentence_end]
            if char in '.!?\n':
                sentence_end += 1
                break
            sentence_end += 1

        return (sentence_start, sentence_end)


class BeamScorer:
    """
    Score and select/merge beams based on quality metrics.

    Implements the scoring function:
    Score = Σ quality(t) + β×Coverage - γ×Redundancy - δ×Length
    """

    def __init__(
        self,
        config: Optional[ScoringConfig] = None,
        triple_scorer: Optional[TripleScorer] = None
    ):
        self.config = config or ScoringConfig()
        self.triple_scorer = triple_scorer or TripleScorer(config)

    def score_beam(
        self,
        statements: list[Statement],
        source_text: str
    ) -> float:
        """
        Compute beam score using the quality formula.

        Score = Σ quality(t) + β×Coverage - γ×Redundancy - δ×Length
        """
        if not statements:
            return 0.0

        # Sum of quality scores
        quality_sum = sum(
            (stmt.confidence_score or self.triple_scorer.score_triple(stmt, source_text))
            for stmt in statements
        )
        quality_term = self.config.quality_weight * quality_sum

        # Coverage bonus
        coverage = self.compute_coverage(statements, source_text)
        coverage_term = self.config.coverage_weight * coverage

        # Redundancy penalty
        redundancy = self.compute_redundancy(statements)
        redundancy_term = self.config.redundancy_penalty * redundancy

        # Length penalty (normalized by statement count)
        length = len(statements)
        length_term = self.config.length_penalty * (length / 10.0)  # Normalize

        return quality_term + coverage_term - redundancy_term - length_term

    def compute_coverage(
        self,
        statements: list[Statement],
        source_text: str
    ) -> float:
        """
        Compute coverage: % of source text tokens explained by evidence spans.
        """
        if not source_text or not statements:
            return 0.0

        # Track which character positions are covered
        covered = set()

        for stmt in statements:
            span = stmt.evidence_span
            if span is None:
                span = self.triple_scorer.find_evidence_span(stmt, source_text)

            if span:
                for i in range(span[0], min(span[1], len(source_text))):
                    covered.add(i)

        # Calculate coverage as percentage of non-whitespace characters
        content_chars = sum(1 for c in source_text if not c.isspace())
        covered_content = sum(1 for i in covered if not source_text[i].isspace())

        return covered_content / content_chars if content_chars > 0 else 0.0

    def compute_redundancy(self, statements: list[Statement]) -> float:
        """
        Compute redundancy penalty for near-duplicate triples.

        Only counts exact duplicates (same subject, predicate, and object).
        Note: Same subject+predicate with different objects is NOT redundant,
        as it represents distinct relationships (e.g., "Apple announced iPhone and iPad").
        """
        if len(statements) < 2:
            return 0.0

        redundant_pairs = 0
        total_pairs = 0

        for i, stmt1 in enumerate(statements):
            for stmt2 in statements[i + 1:]:
                total_pairs += 1

                # Only count exact duplicates (same subject, predicate, AND object)
                if (stmt1.subject.text.lower() == stmt2.subject.text.lower() and
                    stmt1.predicate.lower() == stmt2.predicate.lower() and
                    stmt1.object.text.lower() == stmt2.object.text.lower()):
                    redundant_pairs += 1

        return redundant_pairs / total_pairs if total_pairs > 0 else 0.0

    def score_and_rank_statements(
        self,
        statements: list[Statement],
        source_text: str
    ) -> list[Statement]:
        """
        Score each statement and return sorted by confidence (descending).
        """
        for stmt in statements:
            if stmt.confidence_score is None:
                stmt.confidence_score = self.triple_scorer.score_triple(stmt, source_text)
            if stmt.evidence_span is None:
                stmt.evidence_span = self.triple_scorer.find_evidence_span(stmt, source_text)

        return sorted(statements, key=lambda s: s.confidence_score or 0.0, reverse=True)

    def select_best_beam(
        self,
        candidates: list[list[Statement]],
        source_text: str
    ) -> list[Statement]:
        """
        Select the highest-scoring beam from candidates.
        """
        if not candidates:
            return []

        # Score each candidate and add confidence scores
        scored_candidates = []
        for beam in candidates:
            # Score individual statements
            for stmt in beam:
                if stmt.confidence_score is None:
                    stmt.confidence_score = self.triple_scorer.score_triple(stmt, source_text)
                if stmt.evidence_span is None:
                    stmt.evidence_span = self.triple_scorer.find_evidence_span(stmt, source_text)

            beam_score = self.score_beam(beam, source_text)
            scored_candidates.append((beam_score, beam))

        # Select best
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        return scored_candidates[0][1]

    def merge_beams(
        self,
        candidates: list[list[Statement]],
        source_text: str,
        top_n: Optional[int] = None
    ) -> list[Statement]:
        """
        Merge top-N beams, keeping high-quality unique triples.

        1. Score all beams
        2. Take top N beams
        3. Pool all triples
        4. Filter by confidence threshold
        5. Deduplicate (keeping highest confidence)
        6. Resolve conflicts
        """
        if not candidates:
            return []

        top_n = top_n or self.config.merge_top_n

        # Score each beam
        scored_beams = []
        for beam in candidates:
            for stmt in beam:
                if stmt.confidence_score is None:
                    stmt.confidence_score = self.triple_scorer.score_triple(stmt, source_text)
                if stmt.evidence_span is None:
                    stmt.evidence_span = self.triple_scorer.find_evidence_span(stmt, source_text)

            beam_score = self.score_beam(beam, source_text)
            scored_beams.append((beam_score, beam))

        # Sort and take top N
        scored_beams.sort(key=lambda x: x[0], reverse=True)
        top_beams = [beam for _, beam in scored_beams[:top_n]]

        # Pool all triples
        all_statements: list[Statement] = []
        for beam in top_beams:
            all_statements.extend(beam)

        # Filter by confidence threshold
        min_conf = self.config.min_confidence
        filtered = [s for s in all_statements if (s.confidence_score or 0) >= min_conf]

        # Filter out statements where source_text doesn't support the predicate
        # This catches model hallucinations where predicate doesn't match the evidence
        consistent = [
            s for s in filtered
            if self._source_text_supports_predicate(s)
        ]

        # Deduplicate - keep highest confidence for each (subject, predicate, object)
        # Note: Same subject+predicate with different objects is valid (e.g., "Apple announced X and Y")
        seen: dict[tuple[str, str, str], Statement] = {}
        for stmt in consistent:
            key = (
                stmt.subject.text.lower(),
                stmt.predicate.lower(),
                stmt.object.text.lower()
            )
            if key not in seen or (stmt.confidence_score or 0) > (seen[key].confidence_score or 0):
                seen[key] = stmt

        return list(seen.values())

    def _source_text_supports_predicate(self, stmt: Statement) -> bool:
        """
        Check if a statement's source_text contains a lexical trigger for its predicate.

        Returns True if:
        - source_text is None (no requirement to check)
        - source_text contains at least one significant word from the predicate

        Returns False if:
        - source_text is set but contains no words from the predicate
        """
        if not stmt.source_text:
            return True  # No source_text to check

        predicate_words = stmt.predicate.lower().split()
        source_lower = stmt.source_text.lower()

        # Check if any significant predicate word appears in source_text
        for word in predicate_words:
            if len(word) > 2 and word in source_lower:
                return True

        return False

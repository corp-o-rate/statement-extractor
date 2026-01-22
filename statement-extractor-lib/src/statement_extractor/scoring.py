"""
Scoring module for statement extraction quality assessment.

Provides:
- TripleScorer: Score individual triples combining semantic similarity and grammatical accuracy
- BeamScorer: Score and select/merge beams based on quality metrics
"""

import logging
from typing import Optional

import numpy as np

from .models import ScoringConfig, Statement

logger = logging.getLogger(__name__)


class TripleScorer:
    """
    Score individual triples combining semantic similarity and entity recognition.

    The score is a weighted combination of:
    - Semantic similarity (50%): Cosine similarity between source text and reassembled triple
    - Subject entity score (25%): How entity-like the subject is (via GLiNER2)
    - Object entity score (25%): How entity-like the object is (via GLiNER2)

    Entity scoring (via GLiNER2):
    - Recognized entity with high confidence: 1.0
    - Recognized entity with moderate confidence: 0.8
    - Partially recognized: 0.6
    - Not recognized: 0.2
    """

    def __init__(
        self,
        config: Optional[ScoringConfig] = None,
        device: Optional[str] = None,
    ):
        self.config = config or ScoringConfig()

        # Auto-detect device
        if device is None:
            import torch
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        # Lazy-loaded embedding model
        self._model = None
        self._embedding_model_name = "all-MiniLM-L6-v2"

    def _load_model(self):
        """Load sentence-transformers model lazily."""
        if self._model is not None:
            return

        from sentence_transformers import SentenceTransformer

        logger.debug(f"Loading embedding model: {self._embedding_model_name} on {self.device}")
        self._model = SentenceTransformer(self._embedding_model_name, device=self.device)
        logger.debug(f"Embedding model loaded on {self.device}")

    def _compute_embeddings(self, texts: list[str]) -> np.ndarray:
        """Compute embeddings for a list of texts."""
        self._load_model()
        return self._model.encode(texts, convert_to_numpy=True)

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        dot = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(dot / (norm1 * norm2))

    def _score_noun_content(self, text: str) -> float:
        """
        Score how entity-like a text is using GLiNER2 entity recognition.

        Returns:
            1.0 - Recognized as a named entity with high confidence
            0.8 - Recognized as an entity with moderate confidence
            0.6 - Partially recognized or contains entity-like content
            0.2 - Not recognized as any entity type
        """
        if not text or not text.strip():
            return 0.2

        try:
            from .gliner_extraction import score_entity_content
            return score_entity_content(text)
        except Exception as e:
            logger.debug(f"Entity scoring failed for '{text}': {e}")
            return 0.5  # Neutral score on error

    def score_triple(self, statement: Statement, source_text: str) -> float:
        """
        Score a triple's quality (0-1) combining semantic similarity and grammatical accuracy.

        The score is a weighted combination of:
        - Semantic similarity (50%): How well the triple captures the source meaning
        - Subject noun score (25%): Grammatical quality of subject
        - Object noun score (25%): Grammatical quality of object

        Higher scores indicate better overall quality.
        """
        # Use statement's source_text if available, otherwise use provided source_text
        reference_text = statement.source_text or source_text
        if not reference_text:
            logger.debug(f"  No source text, returning neutral score 0.5")
            return 0.5  # Neutral score if no source text

        # Reassemble the triple
        reassembled = f"{statement.subject.text} {statement.predicate} {statement.object.text}"

        # Compute semantic similarity
        embeddings = self._compute_embeddings([reference_text, reassembled])
        semantic_similarity = self._cosine_similarity(embeddings[0], embeddings[1])

        # Compute grammatical scores for subject and object
        subject_noun_score = self._score_noun_content(statement.subject.text)
        object_noun_score = self._score_noun_content(statement.object.text)

        # Weighted combination: 50% semantic, 25% subject, 25% object
        final_score = (
            semantic_similarity * 0.5 +
            subject_noun_score * 0.25 +
            object_noun_score * 0.25
        )

        logger.debug(
            f"  Score for '{statement.subject.text}' --[{statement.predicate}]--> '{statement.object.text}': "
            f"{final_score:.3f} (semantic={semantic_similarity:.2f}, subj_noun={subject_noun_score:.2f}, obj_noun={object_noun_score:.2f})"
        )

        return final_score

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
        logger.debug(f"Merging beams: {len(candidates)} candidates, selecting top {top_n}")

        # Score each beam
        scored_beams = []
        for i, beam in enumerate(candidates):
            logger.debug(f"  Scoring beam {i} ({len(beam)} statements)...")
            for stmt in beam:
                if stmt.confidence_score is None:
                    stmt.confidence_score = self.triple_scorer.score_triple(stmt, source_text)
                if stmt.evidence_span is None:
                    stmt.evidence_span = self.triple_scorer.find_evidence_span(stmt, source_text)

            beam_score = self.score_beam(beam, source_text)
            scored_beams.append((beam_score, beam))
            logger.debug(f"    Beam {i} score: {beam_score:.3f}")

        # Sort and take top N
        scored_beams.sort(key=lambda x: x[0], reverse=True)
        top_beams = [beam for _, beam in scored_beams[:top_n]]
        logger.debug(f"  Selected top {len(top_beams)} beams")

        # Pool all triples
        all_statements: list[Statement] = []
        for beam in top_beams:
            all_statements.extend(beam)
        logger.debug(f"  Pooled {len(all_statements)} statements from top beams")

        # Filter by confidence threshold
        min_conf = self.config.min_confidence
        filtered = [s for s in all_statements if (s.confidence_score or 0) >= min_conf]
        logger.debug(f"  After confidence filter (>={min_conf}): {len(filtered)} statements")

        # Filter out statements where source_text doesn't support the predicate
        # This catches model hallucinations where predicate doesn't match the evidence
        consistent = [
            s for s in filtered
            if self._source_text_supports_predicate(s)
        ]
        logger.debug(f"  After predicate consistency filter: {len(consistent)} statements")

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

        result = list(seen.values())
        logger.debug(f"  After deduplication: {len(result)} unique statements")

        return result

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

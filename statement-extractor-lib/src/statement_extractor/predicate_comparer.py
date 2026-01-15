"""
Embedding-based predicate comparison and normalization.

Uses sentence-transformers for local, offline embedding computation.
Provides semantic similarity for deduplication and taxonomy matching.
"""

import logging
from typing import Optional

import numpy as np

from .models import (
    PredicateComparisonConfig,
    PredicateMatch,
    PredicateTaxonomy,
    Statement,
)

logger = logging.getLogger(__name__)


class EmbeddingDependencyError(Exception):
    """Raised when sentence-transformers is required but not installed."""
    pass


def _check_embedding_dependency():
    """Check if sentence-transformers is installed, raise helpful error if not."""
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        raise EmbeddingDependencyError(
            "Embedding-based comparison requires sentence-transformers.\n\n"
            "Install with:\n"
            "  pip install corp-extractor[embeddings]\n"
            "  or: pip install sentence-transformers\n\n"
            "To disable embeddings, set embedding_dedup=False in ExtractionOptions."
        )


class PredicateComparer:
    """
    Embedding-based predicate comparison and normalization.

    Features:
    - Map extracted predicates to canonical forms from a taxonomy
    - Detect duplicate/similar predicates for deduplication
    - Fully offline using sentence-transformers
    - Lazy model loading to avoid startup cost
    - Caches taxonomy embeddings for efficiency

    Example:
        >>> taxonomy = PredicateTaxonomy(predicates=["acquired", "founded", "works_for"])
        >>> comparer = PredicateComparer(taxonomy=taxonomy)
        >>> match = comparer.match_to_canonical("bought")
        >>> print(match.canonical)  # "acquired"
        >>> print(match.similarity)  # ~0.82
    """

    def __init__(
        self,
        taxonomy: Optional[PredicateTaxonomy] = None,
        config: Optional[PredicateComparisonConfig] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize the predicate comparer.

        Args:
            taxonomy: Optional canonical predicate taxonomy for normalization
            config: Comparison configuration (uses defaults if not provided)
            device: Device to use ('cuda', 'cpu', or None for auto-detect)

        Raises:
            EmbeddingDependencyError: If sentence-transformers is not installed
        """
        _check_embedding_dependency()

        self.taxonomy = taxonomy
        self.config = config or PredicateComparisonConfig()

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

        # Lazy-loaded resources
        self._model = None
        self._taxonomy_embeddings: Optional[np.ndarray] = None

    def _load_model(self):
        """Load sentence-transformers model lazily."""
        if self._model is not None:
            return

        from sentence_transformers import SentenceTransformer

        logger.info(f"Loading embedding model: {self.config.embedding_model} on {self.device}")
        self._model = SentenceTransformer(self.config.embedding_model, device=self.device)
        logger.info(f"Embedding model loaded on {self.device}")

    def _normalize_text(self, text: str) -> str:
        """Normalize text before embedding."""
        if self.config.normalize_text:
            return text.lower().strip()
        return text.strip()

    def _compute_embeddings(self, texts: list[str]) -> np.ndarray:
        """Compute embeddings for a list of texts."""
        self._load_model()
        normalized = [self._normalize_text(t) for t in texts]
        return self._model.encode(normalized, convert_to_numpy=True)

    def _get_taxonomy_embeddings(self) -> np.ndarray:
        """Get or compute cached taxonomy embeddings."""
        if self.taxonomy is None:
            raise ValueError("No taxonomy provided")

        if self._taxonomy_embeddings is None:
            logger.debug(f"Computing embeddings for {len(self.taxonomy.predicates)} taxonomy predicates")
            self._taxonomy_embeddings = self._compute_embeddings(self.taxonomy.predicates)

        return self._taxonomy_embeddings

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        dot = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(dot / (norm1 * norm2))

    def _cosine_similarity_batch(self, vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between a vector and all rows of a matrix."""
        vec_norm = vec / (np.linalg.norm(vec) + 1e-8)
        matrix_norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8
        matrix_normalized = matrix / matrix_norms
        return np.dot(matrix_normalized, vec_norm)

    # =========================================================================
    # Public API
    # =========================================================================

    def match_to_canonical(self, predicate: str) -> PredicateMatch:
        """
        Match a predicate to the closest canonical form in the taxonomy.

        Args:
            predicate: The extracted predicate to match

        Returns:
            PredicateMatch with canonical form and similarity score
        """
        if self.taxonomy is None or len(self.taxonomy.predicates) == 0:
            return PredicateMatch(original=predicate)

        pred_embedding = self._compute_embeddings([predicate])[0]
        taxonomy_embeddings = self._get_taxonomy_embeddings()

        similarities = self._cosine_similarity_batch(pred_embedding, taxonomy_embeddings)
        best_idx = int(np.argmax(similarities))
        best_score = float(similarities[best_idx])

        if best_score >= self.config.similarity_threshold:
            return PredicateMatch(
                original=predicate,
                canonical=self.taxonomy.predicates[best_idx],
                similarity=best_score,
                matched=True,
            )
        else:
            return PredicateMatch(
                original=predicate,
                canonical=None,
                similarity=best_score,
                matched=False,
            )

    def match_batch(self, predicates: list[str]) -> list[PredicateMatch]:
        """
        Match multiple predicates to canonical forms efficiently.

        Uses batch embedding computation for better performance.

        Args:
            predicates: List of predicates to match

        Returns:
            List of PredicateMatch results
        """
        if self.taxonomy is None or len(self.taxonomy.predicates) == 0:
            return [PredicateMatch(original=p) for p in predicates]

        # Batch embedding computation
        pred_embeddings = self._compute_embeddings(predicates)
        taxonomy_embeddings = self._get_taxonomy_embeddings()

        results = []
        for i, predicate in enumerate(predicates):
            similarities = self._cosine_similarity_batch(
                pred_embeddings[i],
                taxonomy_embeddings
            )
            best_idx = int(np.argmax(similarities))
            best_score = float(similarities[best_idx])

            if best_score >= self.config.similarity_threshold:
                results.append(PredicateMatch(
                    original=predicate,
                    canonical=self.taxonomy.predicates[best_idx],
                    similarity=best_score,
                    matched=True,
                ))
            else:
                results.append(PredicateMatch(
                    original=predicate,
                    canonical=None,
                    similarity=best_score,
                    matched=False,
                ))

        return results

    def are_similar(
        self,
        pred1: str,
        pred2: str,
        threshold: Optional[float] = None
    ) -> bool:
        """
        Check if two predicates are semantically similar.

        Args:
            pred1: First predicate
            pred2: Second predicate
            threshold: Similarity threshold (uses config.dedup_threshold if not provided)

        Returns:
            True if predicates are similar above threshold
        """
        embeddings = self._compute_embeddings([pred1, pred2])
        similarity = self._cosine_similarity(embeddings[0], embeddings[1])
        threshold = threshold if threshold is not None else self.config.dedup_threshold
        return similarity >= threshold

    def compute_similarity(self, pred1: str, pred2: str) -> float:
        """
        Compute similarity score between two predicates.

        Args:
            pred1: First predicate
            pred2: Second predicate

        Returns:
            Cosine similarity score (0.0 to 1.0)
        """
        embeddings = self._compute_embeddings([pred1, pred2])
        return self._cosine_similarity(embeddings[0], embeddings[1])

    def deduplicate_statements(
        self,
        statements: list[Statement],
        entity_canonicalizer: Optional[callable] = None,
        detect_reversals: bool = True,
    ) -> list[Statement]:
        """
        Remove duplicate statements using embedding-based predicate comparison.

        Two statements are considered duplicates if:
        - Canonicalized subjects match AND canonicalized objects match, OR
        - Canonicalized subjects match objects (reversed) when detect_reversals=True
        - Predicates are similar (embedding-based)

        When duplicates are found, keeps the statement with better contextualized
        match (comparing "Subject Predicate Object" against source text).

        For reversed duplicates, the correct orientation is determined by comparing
        both "S P O" and "O P S" against source text.

        Args:
            statements: List of Statement objects
            entity_canonicalizer: Optional function to canonicalize entity text
            detect_reversals: Whether to detect reversed duplicates (default True)

        Returns:
            Deduplicated list of statements (keeps best contextualized match)
        """
        logger.debug(f"Embedding deduplication: {len(statements)} statements, detect_reversals={detect_reversals}")

        if len(statements) <= 1:
            return statements

        def canonicalize(text: str) -> str:
            if entity_canonicalizer:
                return entity_canonicalizer(text)
            return text.lower().strip()

        logger.debug("  Computing predicate embeddings...")
        # Compute all predicate embeddings at once for efficiency
        predicates = [s.predicate for s in statements]
        pred_embeddings = self._compute_embeddings(predicates)
        logger.debug(f"  Computed {len(pred_embeddings)} predicate embeddings")

        logger.debug("  Computing contextualized embeddings (S P O)...")
        # Compute contextualized embeddings: "Subject Predicate Object" for each statement
        contextualized_texts = [
            f"{s.subject.text} {s.predicate} {s.object.text}" for s in statements
        ]
        contextualized_embeddings = self._compute_embeddings(contextualized_texts)

        logger.debug("  Computing reversed embeddings (O P S)...")
        # Compute reversed contextualized embeddings: "Object Predicate Subject"
        reversed_texts = [
            f"{s.object.text} {s.predicate} {s.subject.text}" for s in statements
        ]
        reversed_embeddings = self._compute_embeddings(reversed_texts)

        logger.debug("  Computing source text embeddings...")
        # Compute source text embeddings for scoring which duplicate to keep
        source_embeddings = []
        for stmt in statements:
            source_text = stmt.source_text or f"{stmt.subject.text} {stmt.predicate} {stmt.object.text}"
            source_embeddings.append(self._compute_embeddings([source_text])[0])
        logger.debug("  All embeddings computed, starting comparison loop...")

        unique_statements: list[Statement] = []
        unique_pred_embeddings: list[np.ndarray] = []
        unique_context_embeddings: list[np.ndarray] = []
        unique_reversed_embeddings: list[np.ndarray] = []
        unique_source_embeddings: list[np.ndarray] = []
        unique_indices: list[int] = []

        for i, stmt in enumerate(statements):
            subj_canon = canonicalize(stmt.subject.text)
            obj_canon = canonicalize(stmt.object.text)

            duplicate_idx = None
            is_reversed_match = False

            for j, unique_stmt in enumerate(unique_statements):
                unique_subj = canonicalize(unique_stmt.subject.text)
                unique_obj = canonicalize(unique_stmt.object.text)

                # Check direct match: subject->subject, object->object
                direct_match = (subj_canon == unique_subj and obj_canon == unique_obj)

                # Check reversed match: subject->object, object->subject
                reversed_match = (
                    detect_reversals and
                    subj_canon == unique_obj and
                    obj_canon == unique_subj
                )

                if not direct_match and not reversed_match:
                    continue

                # Check predicate similarity
                similarity = self._cosine_similarity(
                    pred_embeddings[i],
                    unique_pred_embeddings[j]
                )
                if similarity >= self.config.dedup_threshold:
                    duplicate_idx = j
                    is_reversed_match = reversed_match and not direct_match
                    match_type = "reversed" if is_reversed_match else "direct"
                    logger.debug(
                        f"  [{i}] DUPLICATE of [{unique_indices[j]}] ({match_type}, sim={similarity:.3f}): "
                        f"'{stmt.subject.text}' --[{stmt.predicate}]--> '{stmt.object.text}'"
                    )
                    break

            if duplicate_idx is None:
                logger.debug(
                    f"  [{i}] UNIQUE: '{stmt.subject.text}' --[{stmt.predicate}]--> '{stmt.object.text}'"
                )
                # Not a duplicate - add to unique list
                unique_statements.append(stmt)
                unique_pred_embeddings.append(pred_embeddings[i])
                unique_context_embeddings.append(contextualized_embeddings[i])
                unique_reversed_embeddings.append(reversed_embeddings[i])
                unique_source_embeddings.append(source_embeddings[i])
                unique_indices.append(i)
            else:
                existing_stmt = unique_statements[duplicate_idx]

                if is_reversed_match:
                    # Reversed duplicate - determine correct orientation using source text
                    # Compare current's normal vs reversed against its source
                    current_normal_score = self._cosine_similarity(
                        contextualized_embeddings[i], source_embeddings[i]
                    )
                    current_reversed_score = self._cosine_similarity(
                        reversed_embeddings[i], source_embeddings[i]
                    )
                    # Compare existing's normal vs reversed against its source
                    existing_normal_score = self._cosine_similarity(
                        unique_context_embeddings[duplicate_idx],
                        unique_source_embeddings[duplicate_idx]
                    )
                    existing_reversed_score = self._cosine_similarity(
                        unique_reversed_embeddings[duplicate_idx],
                        unique_source_embeddings[duplicate_idx]
                    )

                    # Determine best orientation for current
                    current_best = max(current_normal_score, current_reversed_score)
                    current_should_reverse = current_reversed_score > current_normal_score

                    # Determine best orientation for existing
                    existing_best = max(existing_normal_score, existing_reversed_score)
                    existing_should_reverse = existing_reversed_score > existing_normal_score

                    if current_best > existing_best:
                        # Current is better - use it (possibly reversed)
                        if current_should_reverse:
                            best_stmt = stmt.reversed()
                        else:
                            best_stmt = stmt
                        # Merge entity types from existing (accounting for reversal)
                        if existing_should_reverse:
                            best_stmt = best_stmt.merge_entity_types_from(existing_stmt.reversed())
                        else:
                            best_stmt = best_stmt.merge_entity_types_from(existing_stmt)
                        unique_statements[duplicate_idx] = best_stmt
                        unique_pred_embeddings[duplicate_idx] = pred_embeddings[i]
                        unique_context_embeddings[duplicate_idx] = contextualized_embeddings[i]
                        unique_reversed_embeddings[duplicate_idx] = reversed_embeddings[i]
                        unique_source_embeddings[duplicate_idx] = source_embeddings[i]
                        unique_indices[duplicate_idx] = i
                    else:
                        # Existing is better - possibly fix its orientation
                        if existing_should_reverse and not existing_stmt.was_reversed:
                            best_stmt = existing_stmt.reversed()
                        else:
                            best_stmt = existing_stmt
                        # Merge entity types from current (accounting for reversal)
                        if current_should_reverse:
                            best_stmt = best_stmt.merge_entity_types_from(stmt.reversed())
                        else:
                            best_stmt = best_stmt.merge_entity_types_from(stmt)
                        unique_statements[duplicate_idx] = best_stmt
                else:
                    # Direct duplicate - keep the one with better contextualized match
                    current_score = self._cosine_similarity(
                        contextualized_embeddings[i], source_embeddings[i]
                    )
                    existing_score = self._cosine_similarity(
                        unique_context_embeddings[duplicate_idx],
                        unique_source_embeddings[duplicate_idx]
                    )

                    if current_score > existing_score:
                        # Current statement is a better match - replace
                        merged_stmt = stmt.merge_entity_types_from(existing_stmt)
                        unique_statements[duplicate_idx] = merged_stmt
                        unique_pred_embeddings[duplicate_idx] = pred_embeddings[i]
                        unique_context_embeddings[duplicate_idx] = contextualized_embeddings[i]
                        unique_reversed_embeddings[duplicate_idx] = reversed_embeddings[i]
                        unique_source_embeddings[duplicate_idx] = source_embeddings[i]
                        unique_indices[duplicate_idx] = i
                    else:
                        # Existing statement is better - merge entity types from current
                        merged_stmt = existing_stmt.merge_entity_types_from(stmt)
                        unique_statements[duplicate_idx] = merged_stmt

        logger.debug(f"  Deduplication complete: {len(statements)} -> {len(unique_statements)} statements")
        return unique_statements

    def normalize_predicates(
        self,
        statements: list[Statement]
    ) -> list[Statement]:
        """
        Normalize all predicates in statements to canonical forms.

        Uses contextualized matching: compares "Subject CanonicalPredicate Object"
        against the statement's source text for better semantic matching.

        Sets canonical_predicate field on each statement if a match is found.

        Args:
            statements: List of Statement objects

        Returns:
            Statements with canonical_predicate field populated
        """
        if self.taxonomy is None or len(self.taxonomy.predicates) == 0:
            return statements

        for stmt in statements:
            match = self._match_predicate_contextualized(stmt)
            if match.matched and match.canonical:
                stmt.canonical_predicate = match.canonical

        return statements

    def _match_predicate_contextualized(self, statement: Statement) -> PredicateMatch:
        """
        Match a statement's predicate to canonical form using full context.

        Compares "Subject CanonicalPredicate Object" strings against the
        statement's source text for better semantic matching.

        Args:
            statement: The statement to match

        Returns:
            PredicateMatch with best canonical form
        """
        if self.taxonomy is None or len(self.taxonomy.predicates) == 0:
            return PredicateMatch(original=statement.predicate)

        # Get the reference text to compare against
        # Use source_text if available, otherwise construct from components
        reference_text = statement.source_text or f"{statement.subject.text} {statement.predicate} {statement.object.text}"

        # Compute embedding for the reference text
        reference_embedding = self._compute_embeddings([reference_text])[0]

        # Construct contextualized strings for each canonical predicate
        # Format: "Subject CanonicalPredicate Object"
        canonical_statements = [
            f"{statement.subject.text} {canonical_pred} {statement.object.text}"
            for canonical_pred in self.taxonomy.predicates
        ]

        # Compute embeddings for all canonical statement forms
        canonical_embeddings = self._compute_embeddings(canonical_statements)

        # Find best match
        similarities = self._cosine_similarity_batch(reference_embedding, canonical_embeddings)
        best_idx = int(np.argmax(similarities))
        best_score = float(similarities[best_idx])

        if best_score >= self.config.similarity_threshold:
            return PredicateMatch(
                original=statement.predicate,
                canonical=self.taxonomy.predicates[best_idx],
                similarity=best_score,
                matched=True,
            )
        else:
            return PredicateMatch(
                original=statement.predicate,
                canonical=None,
                similarity=best_score,
                matched=False,
            )

    def detect_and_fix_reversals(
        self,
        statements: list[Statement],
        threshold: float = 0.05,
    ) -> list[Statement]:
        """
        Detect and fix subject-object reversals using embedding comparison.

        For each statement, compares:
        - "Subject Predicate Object" embedding against source_text
        - "Object Predicate Subject" embedding against source_text

        If the reversed version has significantly higher similarity to the source,
        the subject and object are swapped and was_reversed is set to True.

        Args:
            statements: List of Statement objects
            threshold: Minimum similarity difference to trigger reversal (default 0.05)

        Returns:
            List of statements with reversals corrected
        """
        if not statements:
            return statements

        result = []
        for stmt in statements:
            # Skip if no source_text to compare against
            if not stmt.source_text:
                result.append(stmt)
                continue

            # Build normal and reversed triple strings
            normal_text = f"{stmt.subject.text} {stmt.predicate} {stmt.object.text}"
            reversed_text = f"{stmt.object.text} {stmt.predicate} {stmt.subject.text}"

            # Compute embeddings for normal, reversed, and source
            embeddings = self._compute_embeddings([normal_text, reversed_text, stmt.source_text])
            normal_emb, reversed_emb, source_emb = embeddings[0], embeddings[1], embeddings[2]

            # Compute similarities to source
            normal_sim = self._cosine_similarity(normal_emb, source_emb)
            reversed_sim = self._cosine_similarity(reversed_emb, source_emb)

            # If reversed is significantly better, swap subject and object
            if reversed_sim > normal_sim + threshold:
                result.append(stmt.reversed())
            else:
                result.append(stmt)

        return result

    def check_reversal(self, statement: Statement) -> tuple[bool, float, float]:
        """
        Check if a single statement should be reversed.

        Args:
            statement: Statement to check

        Returns:
            Tuple of (should_reverse, normal_similarity, reversed_similarity)
        """
        if not statement.source_text:
            return (False, 0.0, 0.0)

        normal_text = f"{statement.subject.text} {statement.predicate} {statement.object.text}"
        reversed_text = f"{statement.object.text} {statement.predicate} {statement.subject.text}"

        embeddings = self._compute_embeddings([normal_text, reversed_text, statement.source_text])
        normal_emb, reversed_emb, source_emb = embeddings[0], embeddings[1], embeddings[2]

        normal_sim = self._cosine_similarity(normal_emb, source_emb)
        reversed_sim = self._cosine_similarity(reversed_emb, source_emb)

        return (reversed_sim > normal_sim, normal_sim, reversed_sim)

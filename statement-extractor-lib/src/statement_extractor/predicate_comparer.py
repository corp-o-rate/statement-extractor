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
    ):
        """
        Initialize the predicate comparer.

        Args:
            taxonomy: Optional canonical predicate taxonomy for normalization
            config: Comparison configuration (uses defaults if not provided)

        Raises:
            EmbeddingDependencyError: If sentence-transformers is not installed
        """
        _check_embedding_dependency()

        self.taxonomy = taxonomy
        self.config = config or PredicateComparisonConfig()

        # Lazy-loaded resources
        self._model = None
        self._taxonomy_embeddings: Optional[np.ndarray] = None

    def _load_model(self):
        """Load sentence-transformers model lazily."""
        if self._model is not None:
            return

        from sentence_transformers import SentenceTransformer

        logger.info(f"Loading embedding model: {self.config.embedding_model}")
        self._model = SentenceTransformer(self.config.embedding_model)
        logger.info("Embedding model loaded")

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
    ) -> list[Statement]:
        """
        Remove duplicate statements using embedding-based predicate comparison.

        Two statements are considered duplicates if:
        - Canonicalized subjects match
        - Predicates are similar (embedding-based)
        - Canonicalized objects match

        When duplicates are found, keeps the statement with better contextualized
        match (comparing "Subject Predicate Object" against source text).

        Args:
            statements: List of Statement objects
            entity_canonicalizer: Optional function to canonicalize entity text

        Returns:
            Deduplicated list of statements (keeps best contextualized match)
        """
        if len(statements) <= 1:
            return statements

        def canonicalize(text: str) -> str:
            if entity_canonicalizer:
                return entity_canonicalizer(text)
            return text.lower().strip()

        # Compute all predicate embeddings at once for efficiency
        predicates = [s.predicate for s in statements]
        pred_embeddings = self._compute_embeddings(predicates)

        # Compute contextualized embeddings: "Subject Predicate Object" for each statement
        contextualized_texts = [
            f"{s.subject.text} {s.predicate} {s.object.text}" for s in statements
        ]
        contextualized_embeddings = self._compute_embeddings(contextualized_texts)

        # Compute source text embeddings for scoring which duplicate to keep
        source_embeddings = []
        for stmt in statements:
            source_text = stmt.source_text or f"{stmt.subject.text} {stmt.predicate} {stmt.object.text}"
            source_embeddings.append(self._compute_embeddings([source_text])[0])

        unique_statements: list[Statement] = []
        unique_pred_embeddings: list[np.ndarray] = []
        unique_context_embeddings: list[np.ndarray] = []
        unique_source_embeddings: list[np.ndarray] = []
        unique_indices: list[int] = []

        for i, stmt in enumerate(statements):
            subj_canon = canonicalize(stmt.subject.text)
            obj_canon = canonicalize(stmt.object.text)

            duplicate_idx = None

            for j, unique_stmt in enumerate(unique_statements):
                unique_subj = canonicalize(unique_stmt.subject.text)
                unique_obj = canonicalize(unique_stmt.object.text)

                # Check subject and object match
                if subj_canon != unique_subj or obj_canon != unique_obj:
                    continue

                # Check predicate similarity
                similarity = self._cosine_similarity(
                    pred_embeddings[i],
                    unique_pred_embeddings[j]
                )
                if similarity >= self.config.dedup_threshold:
                    duplicate_idx = j
                    break

            if duplicate_idx is None:
                # Not a duplicate - add to unique list
                unique_statements.append(stmt)
                unique_pred_embeddings.append(pred_embeddings[i])
                unique_context_embeddings.append(contextualized_embeddings[i])
                unique_source_embeddings.append(source_embeddings[i])
                unique_indices.append(i)
            else:
                # Duplicate found - keep the one with better contextualized match
                # Compare "Subject Predicate Object" against source text
                current_score = self._cosine_similarity(
                    contextualized_embeddings[i],
                    source_embeddings[i]
                )
                existing_score = self._cosine_similarity(
                    unique_context_embeddings[duplicate_idx],
                    unique_source_embeddings[duplicate_idx]
                )

                if current_score > existing_score:
                    # Current statement is a better match - replace
                    unique_statements[duplicate_idx] = stmt
                    unique_pred_embeddings[duplicate_idx] = pred_embeddings[i]
                    unique_context_embeddings[duplicate_idx] = contextualized_embeddings[i]
                    unique_source_embeddings[duplicate_idx] = source_embeddings[i]
                    unique_indices[duplicate_idx] = i

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

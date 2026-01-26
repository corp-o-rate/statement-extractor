"""
StatementDeduplicator - Hash-based deduplication for statements.

Removes duplicate statements across chunks using normalized hashing.
Works with Stage 2+ output (PipelineStatement, LabeledStatement) which
have subject-predicate-object structure.
"""

import hashlib
import logging
from typing import TypeVar, Union

from ..models.labels import LabeledStatement
from ..models.statement import PipelineStatement

logger = logging.getLogger(__name__)

# Type variable for generic deduplication
T = TypeVar("T", PipelineStatement, LabeledStatement)


class StatementDeduplicator:
    """
    Deduplicates statements using normalized hash comparison.

    Uses a hash of normalized (subject, predicate, object) to identify
    duplicates. Keeps the first occurrence of each unique statement.

    Works with PipelineStatement (Stage 2) and LabeledStatement (Stage 4).
    """

    def __init__(self):
        """Initialize the deduplicator."""
        self._seen_hashes: set[str] = set()

    def reset(self) -> None:
        """Reset the deduplicator state, clearing all seen hashes."""
        self._seen_hashes.clear()
        logger.debug("Deduplicator state reset")

    def _normalize_text(self, text: str) -> str:
        """
        Normalize text for comparison.

        - Lowercase
        - Strip whitespace
        - Collapse multiple spaces
        """
        return " ".join(text.lower().strip().split())

    def _get_triple_parts(
        self,
        stmt: Union[PipelineStatement, LabeledStatement],
    ) -> tuple[str, str, str]:
        """
        Extract (subject, predicate, object) from a statement.

        Handles different statement types consistently.
        """
        if isinstance(stmt, LabeledStatement):
            return (
                stmt.statement.subject.text,
                stmt.statement.predicate,
                stmt.statement.object.text,
            )
        else:
            # PipelineStatement
            return (
                stmt.subject.text,
                stmt.predicate,
                stmt.object.text,
            )

    def _hash_triple(
        self,
        stmt: Union[PipelineStatement, LabeledStatement],
    ) -> str:
        """
        Generate a hash for a statement triple.

        Uses normalized text to catch near-duplicates with different
        casing or whitespace.
        """
        subj, pred, obj = self._get_triple_parts(stmt)

        key = (
            self._normalize_text(subj),
            self._normalize_text(pred),
            self._normalize_text(obj),
        )

        # Use sha256 and truncate to 16 chars for reasonable uniqueness
        return hashlib.sha256(str(key).encode()).hexdigest()[:16]

    def is_duplicate(
        self,
        stmt: Union[PipelineStatement, LabeledStatement],
    ) -> bool:
        """
        Check if a statement is a duplicate.

        Also marks the statement as seen if it's not a duplicate.

        Args:
            stmt: Statement to check

        Returns:
            True if this is a duplicate of a previously seen statement
        """
        hash_value = self._hash_triple(stmt)

        if hash_value in self._seen_hashes:
            return True

        self._seen_hashes.add(hash_value)
        return False

    def filter_duplicates(self, statements: list[T]) -> list[T]:
        """
        Filter out duplicate statements from a list.

        Preserves order and keeps the first occurrence of each unique statement.

        Args:
            statements: List of statements to deduplicate

        Returns:
            List with duplicates removed
        """
        if not statements:
            return []

        original_count = len(statements)
        result = []

        for stmt in statements:
            if not self.is_duplicate(stmt):
                result.append(stmt)

        removed = original_count - len(result)
        if removed > 0:
            logger.info(f"Deduplication removed {removed} statements ({len(result)} remaining)")

        return result

    def deduplicate_batch(
        self,
        statements: list[T],
        reset_first: bool = True,
    ) -> list[T]:
        """
        Deduplicate a batch of statements.

        Optionally resets state before processing to ensure clean deduplication.

        Args:
            statements: List of statements to deduplicate
            reset_first: Whether to reset seen hashes before processing

        Returns:
            Deduplicated list of statements
        """
        if reset_first:
            self.reset()

        return self.filter_duplicates(statements)

    @property
    def seen_count(self) -> int:
        """Get the number of unique statements seen."""
        return len(self._seen_hashes)

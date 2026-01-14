"""
Entity canonicalization for statement deduplication.

Provides default canonicalization functions and a Canonicalizer class
for normalizing entity text before comparison.
"""

import re
from typing import Callable, Optional

from .models import Statement


# Common determiners to remove from the start of entity text
DETERMINERS = frozenset(["the", "a", "an", "this", "that", "these", "those"])


def default_entity_canonicalizer(text: str) -> str:
    """
    Default entity canonicalization function.

    Transformations:
    - Trim leading/trailing whitespace
    - Convert to lowercase
    - Remove leading determiners (the, a, an, etc.)
    - Normalize internal whitespace (multiple spaces -> single)

    Args:
        text: The entity text to canonicalize

    Returns:
        Canonicalized text

    Example:
        >>> default_entity_canonicalizer("  The  Apple Inc.  ")
        'apple inc.'
        >>> default_entity_canonicalizer("A new product")
        'new product'
    """
    # Trim and lowercase
    result = text.strip().lower()

    # Normalize internal whitespace
    result = re.sub(r'\s+', ' ', result)

    # Remove leading determiners
    words = result.split()
    if words and words[0] in DETERMINERS:
        result = ' '.join(words[1:])

    return result.strip()


class Canonicalizer:
    """
    Canonicalize entities for deduplication.

    Supports custom canonicalization functions for entities.
    Predicate comparison uses embeddings (see PredicateComparer).

    Example:
        >>> canon = Canonicalizer()
        >>> canon.canonicalize_entity("The Apple Inc.")
        'apple inc.'

        >>> # With custom function
        >>> canon = Canonicalizer(entity_fn=lambda x: x.upper())
        >>> canon.canonicalize_entity("Apple Inc.")
        'APPLE INC.'
    """

    def __init__(
        self,
        entity_fn: Optional[Callable[[str], str]] = None,
    ):
        """
        Initialize the canonicalizer.

        Args:
            entity_fn: Custom function to canonicalize entity text.
                       If None, uses default_entity_canonicalizer.
        """
        self.entity_fn = entity_fn or default_entity_canonicalizer

    def canonicalize_entity(self, text: str) -> str:
        """
        Canonicalize an entity string.

        Args:
            text: Entity text to canonicalize

        Returns:
            Canonicalized text
        """
        return self.entity_fn(text)

    def canonicalize_statement_entities(
        self,
        statement: Statement
    ) -> tuple[str, str]:
        """
        Return canonicalized (subject, object) tuple.

        Note: Predicate comparison uses embeddings, not text canonicalization.

        Args:
            statement: Statement to canonicalize

        Returns:
            Tuple of (canonicalized_subject, canonicalized_object)
        """
        return (
            self.canonicalize_entity(statement.subject.text),
            self.canonicalize_entity(statement.object.text),
        )

    def create_dedup_key(
        self,
        statement: Statement,
        predicate_canonical: Optional[str] = None
    ) -> tuple[str, str, str]:
        """
        Create a deduplication key for a statement.

        For exact-match deduplication (when not using embedding-based comparison).

        Args:
            statement: Statement to create key for
            predicate_canonical: Optional canonical predicate (if taxonomy was used)

        Returns:
            Tuple of (subject, predicate, object) for deduplication
        """
        subj = self.canonicalize_entity(statement.subject.text)
        obj = self.canonicalize_entity(statement.object.text)
        pred = predicate_canonical or statement.predicate.lower().strip()
        return (subj, pred, obj)


def deduplicate_statements_exact(
    statements: list[Statement],
    entity_canonicalizer: Optional[Callable[[str], str]] = None,
    detect_reversals: bool = True,
) -> list[Statement]:
    """
    Deduplicate statements using exact text matching.

    Use this when embedding-based deduplication is disabled.
    When duplicates are found, entity types are merged - specific types
    (ORG, PERSON, etc.) take precedence over UNKNOWN.

    When detect_reversals=True, also detects reversed duplicates where
    subject and object are swapped. The first occurrence determines the
    canonical orientation.

    Args:
        statements: List of statements to deduplicate
        entity_canonicalizer: Optional custom canonicalization function
        detect_reversals: Whether to detect reversed duplicates (default True)

    Returns:
        Deduplicated list with merged entity types
    """
    if len(statements) <= 1:
        return statements

    canonicalizer = Canonicalizer(entity_fn=entity_canonicalizer)

    # Map from dedup key to index in unique list
    seen: dict[tuple[str, str, str], int] = {}
    unique: list[Statement] = []

    for stmt in statements:
        key = canonicalizer.create_dedup_key(stmt)
        # Also compute reversed key (object, predicate, subject)
        reversed_key = (key[2], key[1], key[0])

        if key in seen:
            # Direct duplicate found - merge entity types
            existing_idx = seen[key]
            existing_stmt = unique[existing_idx]
            merged_stmt = existing_stmt.merge_entity_types_from(stmt)
            unique[existing_idx] = merged_stmt
        elif detect_reversals and reversed_key in seen:
            # Reversed duplicate found - merge entity types (accounting for reversal)
            existing_idx = seen[reversed_key]
            existing_stmt = unique[existing_idx]
            # Merge types from the reversed statement
            merged_stmt = existing_stmt.merge_entity_types_from(stmt.reversed())
            unique[existing_idx] = merged_stmt
        else:
            # New unique statement
            seen[key] = len(unique)
            unique.append(stmt)

    return unique

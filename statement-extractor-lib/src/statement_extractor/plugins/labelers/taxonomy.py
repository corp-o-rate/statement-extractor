"""
TaxonomyLabeler - Classifies statements against a large taxonomy using MNLI.

Uses zero-shot classification with MNLI models for taxonomy labeling where
there are too many possible values for simple multi-choice classification.
"""

import json
import logging
from pathlib import Path
from typing import Optional, TypedDict

from ..base import BaseLabelerPlugin, TaxonomySchema, PluginCapability


class TaxonomyEntry(TypedDict):
    """Structure for each taxonomy label entry."""
    description: str
    id: int
    mnli_label: str
    embedding_label: str


from ...pipeline.context import PipelineContext
from ...models import (
    PipelineStatement,
    CanonicalEntity,
    StatementLabel,
)

logger = logging.getLogger(__name__)

# Default taxonomy file location (relative to this module)
DEFAULT_TAXONOMY_PATH = Path(__file__).parent.parent.parent / "data" / "statement_taxonomy.json"

# Default categories to use (all of them)
DEFAULT_CATEGORIES = [
    "environment",
    "society",
    "governance",
    "animals",
    "industry",
    "human_harm",
    "human_benefit",
    "animal_harm",
    "animal_benefit",
    "environment_harm",
    "environment_benefit",
]


class TaxonomyClassifier:
    """
    MNLI-based zero-shot classifier for taxonomy labeling.

    Uses HuggingFace transformers zero-shot-classification pipeline.
    """

    def __init__(
        self,
        model_id: str = "facebook/bart-large-mnli",
        device: Optional[str] = None,
    ):
        """
        Initialize the classifier.

        Args:
            model_id: HuggingFace model ID for MNLI classification
            device: Device to use ('cuda', 'mps', 'cpu', or None for auto)
        """
        self._model_id = model_id
        self._device = device
        self._classifier = None

    def _load_classifier(self):
        """Lazy-load the zero-shot classification pipeline."""
        if self._classifier is not None:
            return

        try:
            from transformers import pipeline
            import torch

            # Auto-detect device if not specified
            device = self._device
            if device is None:
                if torch.cuda.is_available():
                    device = "cuda"
                elif torch.backends.mps.is_available():
                    device = "mps"
                else:
                    device = "cpu"

            logger.info(f"Loading MNLI classifier '{self._model_id}' on {device}...")
            self._classifier = pipeline(
                "zero-shot-classification",
                model=self._model_id,
                device=device if device != "cpu" else -1,
            )
            logger.debug("MNLI classifier loaded")

        except ImportError as e:
            raise ImportError(
                "transformers is required for MNLI classification. "
                "Install with: pip install transformers"
            ) from e

    def classify(
        self,
        text: str,
        candidate_labels: list[str],
        multi_label: bool = False,
    ) -> tuple[str, float]:
        """
        Classify text against candidate labels using MNLI.

        Args:
            text: Text to classify
            candidate_labels: List of possible labels
            multi_label: Whether multiple labels can apply

        Returns:
            Tuple of (best_label, confidence)
        """
        self._load_classifier()

        result = self._classifier(
            text,
            candidate_labels=candidate_labels,
            multi_label=multi_label,
        )

        # Result format: {'sequence': '...', 'labels': [...], 'scores': [...]}
        best_label = result["labels"][0]
        confidence = result["scores"][0]

        return best_label, confidence

    def classify_hierarchical(
        self,
        text: str,
        taxonomy: dict[str, list[str]],
        top_k_categories: int = 3,
    ) -> tuple[str, str, float]:
        """
        Hierarchical classification: first category, then label within category.

        More efficient than flat classification for large taxonomies.

        Args:
            text: Text to classify
            taxonomy: Dict mapping category -> list of labels
            top_k_categories: Number of top categories to consider

        Returns:
            Tuple of (category, label, confidence)
        """
        self._load_classifier()

        # Step 1: Classify into category
        categories = list(taxonomy.keys())
        cat_result = self._classifier(text, candidate_labels=categories)

        # Get top-k categories
        top_categories = cat_result["labels"][:top_k_categories]
        top_cat_scores = cat_result["scores"][:top_k_categories]

        # Step 2: Classify within top categories
        best_label = None
        best_category = None
        best_score = 0.0

        for cat, cat_score in zip(top_categories, top_cat_scores):
            labels = taxonomy[cat]
            if not labels:
                continue

            label_result = self._classifier(text, candidate_labels=labels)
            label = label_result["labels"][0]
            label_score = label_result["scores"][0]

            # Combined score: category confidence * label confidence
            combined_score = cat_score * label_score

            if combined_score > best_score:
                best_score = combined_score
                best_label = label
                best_category = cat

        return best_category, best_label, best_score


class TaxonomyLabeler(BaseLabelerPlugin):
    """
    Labeler that classifies statements against a large taxonomy using MNLI.

    Supports hierarchical classification for efficiency with large taxonomies.
    """

    def __init__(
        self,
        taxonomy_path: Optional[str | Path] = None,
        categories: Optional[list[str]] = None,
        model_id: str = "facebook/bart-large-mnli",
        use_hierarchical: bool = True,
        top_k_categories: int = 3,
        min_confidence: float = 0.3,
    ):
        """
        Initialize the taxonomy labeler.

        Args:
            taxonomy_path: Path to taxonomy JSON file (default: built-in taxonomy)
            categories: List of categories to use (default: all categories)
            model_id: HuggingFace model ID for MNLI classifier
            use_hierarchical: Use hierarchical classification (category then label)
            top_k_categories: Number of top categories to consider in hierarchical mode
            min_confidence: Minimum confidence threshold for returning a label
        """
        self._taxonomy_path = Path(taxonomy_path) if taxonomy_path else DEFAULT_TAXONOMY_PATH
        self._categories = categories or DEFAULT_CATEGORIES
        self._model_id = model_id
        self._use_hierarchical = use_hierarchical
        self._top_k_categories = top_k_categories
        self._min_confidence = min_confidence

        self._taxonomy: Optional[dict[str, dict[str, TaxonomyEntry]]] = None
        self._classifier: Optional[TaxonomyClassifier] = None

    @property
    def name(self) -> str:
        return "taxonomy_labeler"

    @property
    def priority(self) -> int:
        return 60  # Lower priority than embedding taxonomy (use --plugins taxonomy_labeler to enable)

    @property
    def capabilities(self) -> PluginCapability:
        return PluginCapability.LLM_REQUIRED

    @property
    def description(self) -> str:
        return "Classifies statements against a taxonomy using MNLI zero-shot classification"

    @property
    def label_type(self) -> str:
        return "taxonomy"

    @property
    def taxonomy_schema(self) -> TaxonomySchema:
        """Provide taxonomy schema (for documentation/introspection)."""
        taxonomy = self._load_taxonomy()
        # Filter to selected categories
        filtered = {cat: list(labels.keys()) for cat, labels in taxonomy.items() if cat in self._categories}
        return TaxonomySchema(
            label_type=self.label_type,
            values=filtered,
            description="Statement topic classification against corporate ESG taxonomy",
            scope="statement",
        )

    def _load_taxonomy(self) -> dict[str, dict[str, TaxonomyEntry]]:
        """Load taxonomy from JSON file."""
        if self._taxonomy is not None:
            return self._taxonomy

        if not self._taxonomy_path.exists():
            raise FileNotFoundError(f"Taxonomy file not found: {self._taxonomy_path}")

        with open(self._taxonomy_path) as f:
            self._taxonomy = json.load(f)

        logger.debug(f"Loaded taxonomy with {len(self._taxonomy)} categories")
        return self._taxonomy

    def _get_classifier(self) -> TaxonomyClassifier:
        """Get or create the MNLI classifier."""
        if self._classifier is None:
            self._classifier = TaxonomyClassifier(model_id=self._model_id)
        return self._classifier

    def _get_filtered_taxonomy(self) -> dict[str, list[str]]:
        """Get taxonomy filtered to selected categories, with label names only."""
        taxonomy = self._load_taxonomy()
        return {
            cat: list(labels.keys())
            for cat, labels in taxonomy.items()
            if cat in self._categories
        }

    def label(
        self,
        statement: PipelineStatement,
        subject_canonical: CanonicalEntity,
        object_canonical: CanonicalEntity,
        context: PipelineContext,
    ) -> Optional[StatementLabel]:
        """
        Classify statement against the taxonomy.

        Args:
            statement: The statement to label
            subject_canonical: Canonicalized subject
            object_canonical: Canonicalized object
            context: Pipeline context

        Returns:
            StatementLabel with taxonomy classification, or None if below threshold
        """
        # Check for pre-computed classification from extractor
        result = context.get_classification(statement.source_text, self.label_type)
        if result:
            label_value, confidence = result
            if confidence >= self._min_confidence:
                return StatementLabel(
                    label_type=self.label_type,
                    label_value=label_value,
                    confidence=confidence,
                    labeler=self.name,
                )
            return None

        # Run MNLI classification
        try:
            classifier = self._get_classifier()
            taxonomy = self._get_filtered_taxonomy()

            # Text to classify
            text = statement.source_text

            if self._use_hierarchical:
                category, label, confidence = classifier.classify_hierarchical(
                    text,
                    taxonomy,
                    top_k_categories=self._top_k_categories,
                )
                # Include category in label for clarity
                full_label = f"{category}:{label}" if category and label else None
            else:
                # Flat classification across all labels
                all_labels = []
                for labels in taxonomy.values():
                    all_labels.extend(labels)

                label, confidence = classifier.classify(text, all_labels)
                full_label = label

            if full_label and confidence >= self._min_confidence:
                # Get the numeric ID for reproducibility
                label_id = self._get_label_id(category if self._use_hierarchical else None, label)

                return StatementLabel(
                    label_type=self.label_type,
                    label_value=full_label,
                    confidence=confidence,
                    labeler=self.name,
                    metadata={"label_id": label_id, "category": category} if self._use_hierarchical else {"label_id": label_id},
                )

        except Exception as e:
            logger.warning(f"Taxonomy classification failed: {e}")

        return None

    def _get_label_id(self, category: Optional[str], label: str) -> Optional[int]:
        """Get the numeric ID for a label."""
        taxonomy = self._load_taxonomy()

        if category and category in taxonomy:
            entry = taxonomy[category].get(label)
            if entry:
                return entry.get("id")

        # Search all categories for flat classification
        for cat_labels in taxonomy.values():
            if label in cat_labels:
                entry = cat_labels[label]
                return entry.get("id")

        return None


# Allow importing without decorator for testing
TaxonomyLabelerClass = TaxonomyLabeler
TaxonomyClassifierClass = TaxonomyClassifier

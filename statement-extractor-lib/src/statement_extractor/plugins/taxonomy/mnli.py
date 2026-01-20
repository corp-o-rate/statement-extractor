"""
MNLITaxonomyClassifier - Classifies statements using MNLI zero-shot classification.

Uses HuggingFace transformers zero-shot-classification pipeline for taxonomy labeling
where there are too many possible values for simple multi-choice classification.
"""

import json
import logging
from pathlib import Path
from typing import Optional, TypedDict

from ..base import BaseTaxonomyPlugin, TaxonomySchema, PluginCapability


class TaxonomyEntry(TypedDict):
    """Structure for each taxonomy label entry."""
    description: str
    id: int
    mnli_label: str
    embedding_label: str


from ...pipeline.context import PipelineContext
from ...pipeline.registry import PluginRegistry
from ...models import (
    PipelineStatement,
    CanonicalEntity,
    TaxonomyResult,
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


class MNLIClassifier:
    """
    MNLI-based zero-shot classifier for taxonomy labeling.

    Uses HuggingFace transformers zero-shot-classification pipeline.
    """

    def __init__(
        self,
        model_id: str = "facebook/bart-large-mnli",
        device: Optional[str] = None,
    ):
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

    def classify_hierarchical(
        self,
        text: str,
        taxonomy: dict[str, list[str]],
        top_k_categories: int = 3,
        min_score: float = 0.3,
    ) -> list[tuple[str, str, float]]:
        """
        Hierarchical classification: first category, then labels within category.

        Returns all labels above the threshold, not just the best match.

        Args:
            text: Text to classify
            taxonomy: Dict mapping category -> list of labels
            top_k_categories: Number of top categories to consider
            min_score: Minimum combined score to include in results

        Returns:
            List of (category, label, confidence) tuples above threshold
        """
        self._load_classifier()

        categories = list(taxonomy.keys())
        cat_result = self._classifier(text, candidate_labels=categories)

        top_categories = cat_result["labels"][:top_k_categories]
        top_cat_scores = cat_result["scores"][:top_k_categories]

        results: list[tuple[str, str, float]] = []

        for cat, cat_score in zip(top_categories, top_cat_scores):
            labels = taxonomy[cat]
            if not labels:
                continue

            label_result = self._classifier(text, candidate_labels=labels)

            # Get all labels above threshold for this category
            for label, label_score in zip(label_result["labels"], label_result["scores"]):
                combined_score = cat_score * label_score

                if combined_score >= min_score:
                    results.append((cat, label, combined_score))

        # Sort by confidence descending
        results.sort(key=lambda x: x[2], reverse=True)
        return results


@PluginRegistry.taxonomy
class MNLITaxonomyClassifier(BaseTaxonomyPlugin):
    """
    Taxonomy classifier using MNLI zero-shot classification.

    Supports hierarchical classification for efficiency with large taxonomies.
    """

    def __init__(
        self,
        taxonomy_path: Optional[str | Path] = None,
        categories: Optional[list[str]] = None,
        model_id: str = "facebook/bart-large-mnli",
        top_k_categories: int = 3,
        min_confidence: float = 0.3,
    ):
        self._taxonomy_path = Path(taxonomy_path) if taxonomy_path else DEFAULT_TAXONOMY_PATH
        self._categories = categories or DEFAULT_CATEGORIES
        self._model_id = model_id
        self._top_k_categories = top_k_categories
        self._min_confidence = min_confidence

        self._taxonomy: Optional[dict[str, dict[str, TaxonomyEntry]]] = None
        self._classifier: Optional[MNLIClassifier] = None

    @property
    def name(self) -> str:
        return "mnli_taxonomy_classifier"

    @property
    def priority(self) -> int:
        return 50  # Lower priority than embedding (use --plugins mnli_taxonomy_classifier to enable)

    @property
    def capabilities(self) -> PluginCapability:
        return PluginCapability.LLM_REQUIRED

    @property
    def description(self) -> str:
        return "Classifies statements against a taxonomy using MNLI zero-shot classification"

    @property
    def taxonomy_name(self) -> str:
        return "esg_topics"

    @property
    def taxonomy_schema(self) -> TaxonomySchema:
        taxonomy = self._load_taxonomy()
        filtered = {cat: list(labels.keys()) for cat, labels in taxonomy.items() if cat in self._categories}
        return TaxonomySchema(
            label_type="taxonomy",
            values=filtered,
            description="ESG topic classification taxonomy",
            scope="statement",
        )

    @property
    def supported_categories(self) -> list[str]:
        return self._categories.copy()

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

    def _get_classifier(self) -> MNLIClassifier:
        if self._classifier is None:
            self._classifier = MNLIClassifier(model_id=self._model_id)
        return self._classifier

    def _get_filtered_taxonomy(self) -> dict[str, list[str]]:
        taxonomy = self._load_taxonomy()
        return {
            cat: list(labels.keys())
            for cat, labels in taxonomy.items()
            if cat in self._categories
        }

    def classify(
        self,
        statement: PipelineStatement,
        subject_canonical: CanonicalEntity,
        object_canonical: CanonicalEntity,
        context: PipelineContext,
    ) -> list[TaxonomyResult]:
        """Classify statement against the taxonomy using MNLI.

        Returns all labels above the confidence threshold.
        """
        results: list[TaxonomyResult] = []

        try:
            classifier = self._get_classifier()
            taxonomy = self._get_filtered_taxonomy()

            text = statement.source_text

            classifications = classifier.classify_hierarchical(
                text,
                taxonomy,
                top_k_categories=self._top_k_categories,
                min_score=self._min_confidence,
            )

            for category, label, confidence in classifications:
                label_id = self._get_label_id(category, label)

                results.append(TaxonomyResult(
                    taxonomy_name=self.taxonomy_name,
                    category=category,
                    label=label,
                    label_id=label_id,
                    confidence=round(confidence, 4),
                    classifier=self.name,
                ))

        except Exception as e:
            logger.warning(f"MNLI taxonomy classification failed: {e}")

        return results

    def _get_label_id(self, category: str, label: str) -> Optional[int]:
        taxonomy = self._load_taxonomy()
        if category in taxonomy:
            entry = taxonomy[category].get(label)
            if entry:
                return entry.get("id")
        return None


# For testing without decorator
MNLITaxonomyClassifierClass = MNLITaxonomyClassifier

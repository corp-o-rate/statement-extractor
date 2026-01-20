"""
EmbeddingTaxonomyLabeler - Classifies statements using embedding similarity.

Uses sentence-transformers to embed text and compare to pre-computed label
embeddings using cosine similarity with sigmoid calibration.

This is faster than MNLI but may be less accurate for nuanced classification.
"""

import json
import logging
import time
from pathlib import Path
from typing import Optional, TypedDict

import numpy as np


class TaxonomyEntry(TypedDict):
    """Structure for each taxonomy label entry."""
    description: str
    id: int
    mnli_label: str
    embedding_label: str


from ..base import BaseLabelerPlugin, TaxonomySchema, PluginCapability
from ...pipeline.context import PipelineContext
from ...models import (
    PipelineStatement,
    CanonicalEntity,
    StatementLabel,
)

logger = logging.getLogger(__name__)

# Default taxonomy file location (relative to this module)
DEFAULT_TAXONOMY_PATH = Path(__file__).parent.parent.parent / "data" / "statement_taxonomy.json"

# Default categories to use
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


class EmbeddingClassifier:
    """
    Embedding-based classifier using cosine similarity.

    Pre-computes embeddings for all labels and uses dot product
    (on normalized vectors) for fast classification.
    """

    # Calibration parameters to spread out cosine similarity scores
    SIMILARITY_THRESHOLD = 0.65
    CALIBRATION_STEEPNESS = 25.0

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
    ):
        """
        Initialize the classifier.

        Args:
            model_name: sentence-transformers model ID
            device: Device to use ('cuda', 'mps', 'cpu', or None for auto)
        """
        self._model_name = model_name
        self._device = device
        self._model = None

        # Pre-computed label embeddings: {category: {label: embedding}}
        self._label_embeddings: dict[str, dict[str, np.ndarray]] = {}

    def _load_model(self):
        """Lazy-load the embedding model."""
        if self._model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer
            import torch

            # Auto-detect device
            device = self._device
            if device is None:
                if torch.cuda.is_available():
                    device = "cuda"
                elif torch.backends.mps.is_available():
                    device = "mps"
                else:
                    device = "cpu"

            logger.info(f"Loading embedding model '{self._model_name}' on {device}...")
            self._model = SentenceTransformer(self._model_name, device=device)
            logger.debug("Embedding model loaded")

        except ImportError as e:
            raise ImportError(
                "sentence-transformers is required for embedding classification. "
                "Install with: pip install sentence-transformers"
            ) from e

    def precompute_label_embeddings(
        self,
        taxonomy: dict[str, dict[str, TaxonomyEntry]],
        categories: Optional[list[str]] = None,
    ) -> None:
        """
        Pre-compute embeddings for all label names.

        Args:
            taxonomy: Taxonomy dict {category: {label: TaxonomyEntry, ...}, ...}
            categories: Categories to include (default: all)
        """
        self._load_model()

        start_time = time.perf_counter()
        total_labels = 0

        categories_to_process = categories or list(taxonomy.keys())

        for category in categories_to_process:
            if category not in taxonomy:
                continue

            labels = taxonomy[category]
            label_names = list(labels.keys())

            if not label_names:
                continue

            # Batch embed all labels in this category
            embeddings = self._model.encode(label_names, convert_to_numpy=True)

            # Normalize and store
            self._label_embeddings[category] = {}
            for label_name, embedding in zip(label_names, embeddings):
                norm = np.linalg.norm(embedding)
                normalized = embedding / (norm + 1e-8)
                self._label_embeddings[category][label_name] = normalized.astype(np.float32)
                total_labels += 1

        elapsed = time.perf_counter() - start_time
        logger.info(
            f"Pre-computed embeddings for {total_labels} labels "
            f"across {len(self._label_embeddings)} categories in {elapsed:.2f}s"
        )

    def _calibrate_score(self, raw_similarity: float) -> float:
        """
        Apply sigmoid calibration to amplify score differences.

        Cosine similarities cluster in a narrow range (0.5-0.9).
        This transformation spreads them out for better discrimination.
        """
        # Normalize from [-1, 1] to [0, 1]
        normalized = (raw_similarity + 1) / 2

        # Apply sigmoid transformation
        exponent = -self.CALIBRATION_STEEPNESS * (normalized - self.SIMILARITY_THRESHOLD)
        return 1.0 / (1.0 + np.exp(exponent))

    def classify(
        self,
        text: str,
        categories: Optional[list[str]] = None,
        top_k: int = 5,
        min_score: float = 0.3,
    ) -> list[tuple[str, str, float]]:
        """
        Classify text against pre-computed label embeddings.

        Args:
            text: Text to classify
            categories: Categories to search (default: all pre-computed)
            top_k: Number of top results to return
            min_score: Minimum calibrated score threshold

        Returns:
            List of (category, label, score) tuples, sorted by score descending
        """
        self._load_model()

        if not self._label_embeddings:
            raise RuntimeError("Label embeddings not pre-computed. Call precompute_label_embeddings first.")

        # Embed input text
        input_embedding = self._model.encode(text, convert_to_numpy=True)
        input_norm = np.linalg.norm(input_embedding)
        input_normalized = input_embedding / (input_norm + 1e-8)

        # Classify against each category
        categories_to_process = categories or list(self._label_embeddings.keys())
        all_results: list[tuple[str, str, float]] = []

        for category in categories_to_process:
            if category not in self._label_embeddings:
                continue

            for label, label_embedding in self._label_embeddings[category].items():
                # Cosine similarity (both vectors are normalized)
                raw_sim = float(np.dot(input_normalized, label_embedding))
                calibrated_score = self._calibrate_score(raw_sim)

                if calibrated_score >= min_score:
                    all_results.append((category, label, calibrated_score))

        # Sort by score descending and return top-k
        all_results.sort(key=lambda x: x[2], reverse=True)
        return all_results[:top_k]

    def classify_hierarchical(
        self,
        text: str,
        top_k_categories: int = 3,
        top_k_labels: int = 3,
        min_score: float = 0.3,
    ) -> tuple[str, str, float]:
        """
        Hierarchical classification: find best category, then best label.

        More efficient for very large taxonomies.

        Args:
            text: Text to classify
            top_k_categories: Number of top categories to consider
            top_k_labels: Number of labels per category to consider
            min_score: Minimum score threshold

        Returns:
            Tuple of (category, label, score) for best match
        """
        self._load_model()

        if not self._label_embeddings:
            raise RuntimeError("Label embeddings not pre-computed.")

        # Embed input text
        input_embedding = self._model.encode(text, convert_to_numpy=True)
        input_norm = np.linalg.norm(input_embedding)
        input_normalized = input_embedding / (input_norm + 1e-8)

        # First, compute average similarity to each category
        category_scores: list[tuple[str, float]] = []
        for category, labels in self._label_embeddings.items():
            if not labels:
                continue

            # Average similarity to all labels in category
            sims = []
            for label_embedding in labels.values():
                sim = float(np.dot(input_normalized, label_embedding))
                sims.append(sim)

            avg_sim = np.mean(sims)
            category_scores.append((category, avg_sim))

        # Sort categories by average similarity
        category_scores.sort(key=lambda x: x[1], reverse=True)

        # Find best label within top categories
        best_result = (None, None, 0.0)

        for category, _ in category_scores[:top_k_categories]:
            for label, label_embedding in self._label_embeddings[category].items():
                raw_sim = float(np.dot(input_normalized, label_embedding))
                calibrated_score = self._calibrate_score(raw_sim)

                if calibrated_score > best_result[2]:
                    best_result = (category, label, calibrated_score)

        if best_result[0] and best_result[2] >= min_score:
            return best_result

        return (None, None, 0.0)


class EmbeddingTaxonomyLabeler(BaseLabelerPlugin):
    """
    Labeler that classifies statements using embedding similarity.

    Faster than MNLI but may be less accurate for nuanced classification.
    Good for high-throughput scenarios.
    """

    def __init__(
        self,
        taxonomy_path: Optional[str | Path] = None,
        categories: Optional[list[str]] = None,
        model_name: str = "all-MiniLM-L6-v2",
        use_hierarchical: bool = True,
        top_k_categories: int = 3,
        min_confidence: float = 0.3,
    ):
        """
        Initialize the embedding taxonomy labeler.

        Args:
            taxonomy_path: Path to taxonomy JSON file (default: built-in taxonomy)
            categories: List of categories to use (default: all categories)
            model_name: sentence-transformers model ID
            use_hierarchical: Use hierarchical classification for efficiency
            top_k_categories: Number of top categories to consider in hierarchical mode
            min_confidence: Minimum confidence threshold for returning a label
        """
        self._taxonomy_path = Path(taxonomy_path) if taxonomy_path else DEFAULT_TAXONOMY_PATH
        self._categories = categories or DEFAULT_CATEGORIES
        self._model_name = model_name
        self._use_hierarchical = use_hierarchical
        self._top_k_categories = top_k_categories
        self._min_confidence = min_confidence

        self._taxonomy: Optional[dict[str, dict[str, TaxonomyEntry]]] = None
        self._classifier: Optional[EmbeddingClassifier] = None
        self._embeddings_computed = False

    @property
    def name(self) -> str:
        return "embedding_taxonomy_labeler"

    @property
    def priority(self) -> int:
        return 45  # Higher priority than MNLI - default taxonomy labeler (faster)

    @property
    def capabilities(self) -> PluginCapability:
        return PluginCapability.LLM_REQUIRED | PluginCapability.BATCH_PROCESSING

    @property
    def description(self) -> str:
        return "Classifies statements using embedding similarity (faster than MNLI)"

    @property
    def label_type(self) -> str:
        return "taxonomy_embedding"

    @property
    def taxonomy_schema(self) -> TaxonomySchema:
        """Provide taxonomy schema (for documentation/introspection)."""
        taxonomy = self._load_taxonomy()
        filtered = {cat: list(labels.keys()) for cat, labels in taxonomy.items() if cat in self._categories}
        return TaxonomySchema(
            label_type=self.label_type,
            values=filtered,
            description="Statement topic classification using embedding similarity",
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

    def _get_classifier(self) -> EmbeddingClassifier:
        """Get or create the embedding classifier."""
        if self._classifier is None:
            self._classifier = EmbeddingClassifier(model_name=self._model_name)

        if not self._embeddings_computed:
            taxonomy = self._load_taxonomy()
            self._classifier.precompute_label_embeddings(taxonomy, self._categories)
            self._embeddings_computed = True

        return self._classifier

    def label(
        self,
        statement: PipelineStatement,
        subject_canonical: CanonicalEntity,
        object_canonical: CanonicalEntity,
        context: PipelineContext,
    ) -> Optional[StatementLabel]:
        """
        Classify statement using embedding similarity.

        Args:
            statement: The statement to label
            subject_canonical: Canonicalized subject
            object_canonical: Canonicalized object
            context: Pipeline context

        Returns:
            StatementLabel with taxonomy classification, or None if below threshold
        """
        # Check for pre-computed classification
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

        # Run embedding classification
        try:
            classifier = self._get_classifier()
            text = statement.source_text

            if self._use_hierarchical:
                category, label, confidence = classifier.classify_hierarchical(
                    text,
                    top_k_categories=self._top_k_categories,
                    min_score=self._min_confidence,
                )
                if category and label:
                    full_label = f"{category}:{label}"
                else:
                    return None
            else:
                results = classifier.classify(
                    text,
                    top_k=1,
                    min_score=self._min_confidence,
                )
                if results:
                    category, label, confidence = results[0]
                    full_label = f"{category}:{label}"
                else:
                    return None

            # Get the numeric ID for reproducibility
            label_id = self._get_label_id(category, label)

            return StatementLabel(
                label_type=self.label_type,
                label_value=full_label,
                confidence=round(confidence, 4),
                labeler=self.name,
                metadata={"label_id": label_id, "category": category},
            )

        except Exception as e:
            logger.warning(f"Embedding taxonomy classification failed: {e}")

        return None

    def _get_label_id(self, category: str, label: str) -> Optional[int]:
        """Get the numeric ID for a label."""
        taxonomy = self._load_taxonomy()

        if category in taxonomy:
            entry = taxonomy[category].get(label)
            if entry:
                return entry.get("id")

        return None


# Allow importing without decorator for testing
EmbeddingTaxonomyLabelerClass = EmbeddingTaxonomyLabeler
EmbeddingClassifierClass = EmbeddingClassifier

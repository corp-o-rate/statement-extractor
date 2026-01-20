"""
EmbeddingTaxonomyClassifier - Classifies statements using embedding similarity.

Uses sentence-transformers to embed text and compare to pre-computed label
embeddings using cosine similarity with sigmoid calibration.

Faster than MNLI but may be less accurate for nuanced classification.
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

from ..base import BaseTaxonomyPlugin, TaxonomySchema, PluginCapability
from ...pipeline.context import PipelineContext
from ...pipeline.registry import PluginRegistry
from ...models import (
    PipelineStatement,
    CanonicalEntity,
    TaxonomyResult,
)

logger = logging.getLogger(__name__)

# Default taxonomy file location
DEFAULT_TAXONOMY_PATH = Path(__file__).parent.parent.parent / "data" / "statement_taxonomy.json"

# Default categories
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

    SIMILARITY_THRESHOLD = 0.65
    CALIBRATION_STEEPNESS = 25.0

    def __init__(
        self,
        model_name: str = "google/embeddinggemma-300m",
        device: Optional[str] = None,
    ):
        self._model_name = model_name
        self._device = device
        self._model = None
        self._label_embeddings: dict[str, dict[str, np.ndarray]] = {}
        self._text_embedding_cache: dict[str, np.ndarray] = {}  # Cache for input text embeddings

    def _load_model(self):
        if self._model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer
            import torch

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
        """Pre-compute embeddings for all label names."""
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

            embeddings = self._model.encode(label_names, convert_to_numpy=True, show_progress_bar=False)

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
        normalized = (raw_similarity + 1) / 2
        exponent = -self.CALIBRATION_STEEPNESS * (normalized - self.SIMILARITY_THRESHOLD)
        return 1.0 / (1.0 + np.exp(exponent))

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        """
        Encode multiple texts into normalized embeddings in a single batch.

        Uses caching to avoid re-encoding previously seen texts.

        Args:
            texts: List of texts to encode

        Returns:
            2D numpy array of shape (len(texts), embedding_dim) with normalized embeddings
        """
        self._load_model()

        # Separate cached from uncached texts
        uncached_indices = []
        uncached_texts = []
        for i, text in enumerate(texts):
            if text not in self._text_embedding_cache:
                uncached_indices.append(i)
                uncached_texts.append(text)

        # Batch encode uncached texts
        if uncached_texts:
            embeddings = self._model.encode(uncached_texts, convert_to_numpy=True, show_progress_bar=False)
            for i, (text, embedding) in enumerate(zip(uncached_texts, embeddings)):
                norm = np.linalg.norm(embedding)
                normalized = (embedding / (norm + 1e-8)).astype(np.float32)
                self._text_embedding_cache[text] = normalized

            logger.debug(f"Batch encoded {len(uncached_texts)} texts (cache size: {len(self._text_embedding_cache)})")

        # Build result array from cache
        result = np.stack([self._text_embedding_cache[text] for text in texts])
        return result

    def classify_batch(
        self,
        texts: list[str],
        top_k_categories: int = 3,
        min_score: float = 0.3,
    ) -> list[list[tuple[str, str, float]]]:
        """
        Classify multiple texts in a single batch for efficiency.

        Args:
            texts: List of texts to classify
            top_k_categories: Number of top categories to consider per text
            min_score: Minimum calibrated score to include in results

        Returns:
            List of classification results, one list per input text
        """
        if not texts:
            return []

        self._load_model()

        if not self._label_embeddings:
            raise RuntimeError("Label embeddings not pre-computed.")

        # Batch encode all texts
        input_embeddings = self.encode_batch(texts)

        # Prepare label embeddings as matrices for vectorized similarity
        all_results: list[list[tuple[str, str, float]]] = []

        for input_normalized in input_embeddings:
            # Compute average similarity to each category
            category_scores: list[tuple[str, float]] = []
            for category, labels in self._label_embeddings.items():
                if not labels:
                    continue

                sims = []
                for label_embedding in labels.values():
                    sim = float(np.dot(input_normalized, label_embedding))
                    sims.append(sim)

                avg_sim = np.mean(sims)
                category_scores.append((category, avg_sim))

            category_scores.sort(key=lambda x: x[1], reverse=True)

            results: list[tuple[str, str, float]] = []

            for category, _ in category_scores[:top_k_categories]:
                for label, label_embedding in self._label_embeddings[category].items():
                    raw_sim = float(np.dot(input_normalized, label_embedding))
                    calibrated_score = self._calibrate_score(raw_sim)

                    if calibrated_score >= min_score:
                        results.append((category, label, calibrated_score))

            # Sort by confidence descending
            results.sort(key=lambda x: x[2], reverse=True)
            all_results.append(results)

        return all_results

    def classify_hierarchical(
        self,
        text: str,
        top_k_categories: int = 3,
        min_score: float = 0.3,
    ) -> list[tuple[str, str, float]]:
        """Hierarchical classification: find categories, then all labels above threshold.

        Returns all labels above the threshold, not just the best match.

        Args:
            text: Text to classify
            top_k_categories: Number of top categories to consider
            min_score: Minimum calibrated score to include in results

        Returns:
            List of (category, label, confidence) tuples above threshold
        """
        # Use batch method for single text
        results = self.classify_batch([text], top_k_categories, min_score)
        return results[0] if results else []


@PluginRegistry.taxonomy
class EmbeddingTaxonomyClassifier(BaseTaxonomyPlugin):
    """
    Taxonomy classifier using embedding similarity.

    Faster than MNLI, good for high-throughput scenarios.
    """

    def __init__(
        self,
        taxonomy_path: Optional[str | Path] = None,
        categories: Optional[list[str]] = None,
        model_name: str = "google/embeddinggemma-300m",
        top_k_categories: int = 3,
        min_confidence: float = 0.8,
    ):
        self._taxonomy_path = Path(taxonomy_path) if taxonomy_path else DEFAULT_TAXONOMY_PATH
        self._categories = categories or DEFAULT_CATEGORIES
        self._model_name = model_name
        self._top_k_categories = top_k_categories
        self._min_confidence = min_confidence

        self._taxonomy: Optional[dict[str, dict[str, TaxonomyEntry]]] = None
        self._classifier: Optional[EmbeddingClassifier] = None
        self._embeddings_computed = False

    @property
    def name(self) -> str:
        return "embedding_taxonomy_classifier"

    @property
    def priority(self) -> int:
        return 10  # High priority - default taxonomy classifier (faster than MNLI)

    @property
    def capabilities(self) -> PluginCapability:
        return PluginCapability.LLM_REQUIRED | PluginCapability.BATCH_PROCESSING

    @property
    def description(self) -> str:
        return "Classifies statements using embedding similarity (faster than MNLI)"

    @property
    def model_vram_gb(self) -> float:
        """EmbeddingGemma model weights ~1.2GB."""
        return 1.2

    @property
    def per_item_vram_gb(self) -> float:
        """Each text embedding ~0.05GB (embeddings are small)."""
        return 0.05

    @property
    def taxonomy_name(self) -> str:
        return "esg_topics_embedding"

    @property
    def taxonomy_schema(self) -> TaxonomySchema:
        taxonomy = self._load_taxonomy()
        filtered = {cat: list(labels.keys()) for cat, labels in taxonomy.items() if cat in self._categories}
        return TaxonomySchema(
            label_type="taxonomy",
            values=filtered,
            description="ESG topic classification using embeddings",
            scope="statement",
        )

    @property
    def supported_categories(self) -> list[str]:
        return self._categories.copy()

    def _load_taxonomy(self) -> dict[str, dict[str, TaxonomyEntry]]:
        if self._taxonomy is not None:
            return self._taxonomy

        if not self._taxonomy_path.exists():
            raise FileNotFoundError(f"Taxonomy file not found: {self._taxonomy_path}")

        with open(self._taxonomy_path) as f:
            self._taxonomy = json.load(f)

        logger.debug(f"Loaded taxonomy with {len(self._taxonomy)} categories")
        return self._taxonomy

    def _get_classifier(self) -> EmbeddingClassifier:
        if self._classifier is None:
            self._classifier = EmbeddingClassifier(model_name=self._model_name)

        if not self._embeddings_computed:
            taxonomy = self._load_taxonomy()
            self._classifier.precompute_label_embeddings(taxonomy, self._categories)
            self._embeddings_computed = True

        return self._classifier

    def classify(
        self,
        statement: PipelineStatement,
        subject_canonical: CanonicalEntity,
        object_canonical: CanonicalEntity,
        context: PipelineContext,
    ) -> list[TaxonomyResult]:
        """Classify statement using embedding similarity.

        Returns all labels above the confidence threshold.
        """
        results: list[TaxonomyResult] = []

        try:
            classifier = self._get_classifier()
            text = statement.source_text

            classifications = classifier.classify_hierarchical(
                text,
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
            logger.warning(f"Embedding taxonomy classification failed: {e}")

        return results

    def _get_label_id(self, category: str, label: str) -> Optional[int]:
        taxonomy = self._load_taxonomy()
        if category in taxonomy:
            entry = taxonomy[category].get(label)
            if entry:
                return entry.get("id")
        return None

    def classify_batch(
        self,
        items: list[tuple[PipelineStatement, CanonicalEntity, CanonicalEntity]],
        context: PipelineContext,
    ) -> list[list[TaxonomyResult]]:
        """
        Classify multiple statements in a single batch for efficiency.

        Batch encodes all source texts, then classifies each against the taxonomy.

        Args:
            items: List of (statement, subject_canonical, object_canonical) tuples
            context: Pipeline context

        Returns:
            List of TaxonomyResult lists, one per input statement
        """
        if not items:
            return []

        # Extract unique source texts (may have duplicates across statements)
        texts = [stmt.source_text for stmt, _, _ in items]
        unique_texts = list(set(texts))

        logger.info(f"Batch classifying {len(items)} statements ({len(unique_texts)} unique texts)")

        try:
            classifier = self._get_classifier()

            # Batch classify all unique texts
            batch_results = classifier.classify_batch(
                unique_texts,
                top_k_categories=self._top_k_categories,
                min_score=self._min_confidence,
            )

            # Map unique texts to their classifications
            text_to_results: dict[str, list[tuple[str, str, float]]] = {
                text: results for text, results in zip(unique_texts, batch_results)
            }

            # Build results for each input statement
            all_results: list[list[TaxonomyResult]] = []
            for stmt, _, _ in items:
                classifications = text_to_results.get(stmt.source_text, [])

                results: list[TaxonomyResult] = []
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

                all_results.append(results)

            return all_results

        except Exception as e:
            logger.warning(f"Batch taxonomy classification failed: {e}")
            # Return empty results for all items
            return [[] for _ in items]


# For testing without decorator
EmbeddingTaxonomyClassifierClass = EmbeddingTaxonomyClassifier

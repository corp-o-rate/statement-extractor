"""
GLiNER2Extractor - Stage 2 plugin that extracts triples from sentences.

Uses GLiNER2 for:
1. Entity extraction: Identify subject/object entities with types
2. Relation extraction: Extract predicates using predicate list
3. Entity scoring: Score how entity-like subjects/objects are
4. Classification: Run labeler classification schemas in single pass
"""

import json
import logging
from pathlib import Path
from typing import Optional

from ..base import BaseExtractorPlugin, ClassificationSchema, PluginCapability
from ...pipeline.context import PipelineContext
from ...pipeline.registry import PluginRegistry
from ...models import SplitSentence, PipelineStatement, ExtractedEntity, EntityType

logger = logging.getLogger(__name__)

# Type alias for predicate configuration with description and threshold
PredicateConfig = dict[str, str | float]  # {"description": str, "threshold": float}

# Path to bundled default predicates JSON
DEFAULT_PREDICATES_PATH = Path(__file__).parent.parent.parent / "data" / "default_predicates.json"


def load_predicates_from_json(path: Path) -> dict[str, dict[str, PredicateConfig]]:
    """
    Load predicate categories from a JSON file.

    Args:
        path: Path to JSON file containing predicate categories

    Returns:
        Dict of category -> {predicate -> {description, threshold}}

    Raises:
        FileNotFoundError: If path doesn't exist
        json.JSONDecodeError: If JSON is invalid
    """
    with open(path) as f:
        return json.load(f)


def _load_default_predicates() -> dict[str, dict[str, PredicateConfig]]:
    """Load the bundled default predicates."""
    try:
        return load_predicates_from_json(DEFAULT_PREDICATES_PATH)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.warning(f"Failed to load default predicates from {DEFAULT_PREDICATES_PATH}: {e}")
        return {}


# Load default predicates on module import
PREDICATE_CATEGORIES: dict[str, dict[str, PredicateConfig]] = _load_default_predicates()

# Ensure we have predicates loaded
if not PREDICATE_CATEGORIES:
    logger.error("No predicate categories loaded - relation extraction will fail")

# Build reverse lookup: predicate -> category
PREDICATE_TO_CATEGORY: dict[str, str] = {}
for category, predicates in PREDICATE_CATEGORIES.items():
    for predicate in predicates.keys():
        PREDICATE_TO_CATEGORY[predicate] = category



def get_predicate_category(predicate: str) -> Optional[str]:
    """
    Look up the category for a predicate.

    Args:
        predicate: The predicate string to look up

    Returns:
        The category name if found, None otherwise
    """
    # Direct lookup
    if predicate in PREDICATE_TO_CATEGORY:
        return PREDICATE_TO_CATEGORY[predicate]

    # Try normalized form (lowercase, underscores)
    normalized = predicate.lower().replace(" ", "_").replace("-", "_")
    if normalized in PREDICATE_TO_CATEGORY:
        return PREDICATE_TO_CATEGORY[normalized]

    return None


# GLiNER2 entity type to our EntityType mapping
GLINER_TYPE_MAP = {
    "person": EntityType.PERSON,
    "organization": EntityType.ORG,
    "company": EntityType.ORG,
    "location": EntityType.LOC,
    "city": EntityType.GPE,
    "country": EntityType.GPE,
    "product": EntityType.PRODUCT,
    "event": EntityType.EVENT,
    "date": EntityType.DATE,
    "money": EntityType.MONEY,
    "quantity": EntityType.QUANTITY,
}


@PluginRegistry.extractor
class GLiNER2Extractor(BaseExtractorPlugin):
    """
    Extractor plugin that uses GLiNER2 for entity and relation extraction.

    Processes split sentences from Stage 1 and produces PipelineStatement
    objects with subject-predicate-object triples and typed entities.
    Also runs classification schemas from labeler plugins in a single pass.
    """

    def __init__(
        self,
        predicates_file: Optional[str | Path] = None,
        entity_types: Optional[list[str]] = None,
        classification_schemas: Optional[list[ClassificationSchema]] = None,
        min_confidence: float = 0.75,
    ):
        """
        Initialize the GLiNER2 extractor.

        Args:
            predicates_file: Optional path to custom predicates JSON file.
                            If not provided, uses bundled default_predicates.json.
            entity_types: Optional list of entity types to extract.
                         If not provided, uses bundled default from JSON config.
            classification_schemas: Optional list of classification schemas from labelers
            min_confidence: Minimum confidence threshold for relation extraction (default 0.75)
        """
        self._predicates_file = Path(predicates_file) if predicates_file else None
        self._predicate_categories: Optional[dict[str, dict[str, PredicateConfig]]] = None
        self._entity_types = entity_types
        self._classification_schemas = classification_schemas or []
        self._min_confidence = min_confidence
        self._model = None

        # Load custom predicates if file provided
        if self._predicates_file:
            try:
                self._predicate_categories = load_predicates_from_json(self._predicates_file)
                logger.info(f"Loaded {len(self._predicate_categories)} predicate categories from {self._predicates_file}")
            except Exception as e:
                logger.warning(f"Failed to load custom predicates from {self._predicates_file}: {e}")
                self._predicate_categories = None

    def _get_predicate_categories(self) -> dict[str, dict[str, PredicateConfig]]:
        """Get predicate categories - custom file or default from JSON."""
        if self._predicate_categories is not None:
            return self._predicate_categories
        return PREDICATE_CATEGORIES

    def _get_entity_types(self) -> list[str]:
        """Get entity types - from init or derived from GLINER_TYPE_MAP keys."""
        if self._entity_types is not None:
            return self._entity_types
        # Use keys from GLINER_TYPE_MAP as default entity types
        return list(GLINER_TYPE_MAP.keys())

    @property
    def name(self) -> str:
        return "gliner2_extractor"

    @property
    def priority(self) -> int:
        return 10  # High priority - primary extractor

    @property
    def capabilities(self) -> PluginCapability:
        return PluginCapability.BATCH_PROCESSING | PluginCapability.LLM_REQUIRED

    @property
    def description(self) -> str:
        return "GLiNER2 model for entity and relation extraction"

    @property
    def model_vram_gb(self) -> float:
        """GLiNER2 model weights ~0.8GB."""
        return 0.8

    @property
    def per_item_vram_gb(self) -> float:
        """Each triple during batch processing ~0.1GB."""
        return 0.1

    def _get_model(self):
        """Lazy-load the GLiNER2 model."""
        if self._model is None:
            try:
                from gliner2 import GLiNER2
                logger.info("Loading GLiNER2 model...")
                self._model = GLiNER2.from_pretrained("fastino/gliner2-base-v1")
                logger.debug("GLiNER2 model loaded")
            except ImportError:
                logger.warning("GLiNER2 not installed, using fallback")
                self._model = None
        return self._model

    def add_classification_schema(self, schema: ClassificationSchema) -> None:
        """Add a classification schema to run during extraction."""
        self._classification_schemas.append(schema)

    def extract(
        self,
        split_sentences: list[SplitSentence],
        context: PipelineContext,
    ) -> list[PipelineStatement]:
        """
        Extract subject-predicate-object triples from split sentences using GLiNER2.

        Returns ALL matching relations from GLiNER2 (not just the best one).
        Also runs any classification schemas and stores results in context.

        Args:
            split_sentences: Atomic sentences from Stage 1
            context: Pipeline context

        Returns:
            List of PipelineStatement objects (may contain multiple per sentence)
        """
        predicate_categories = self._get_predicate_categories()
        logger.info(f"GLiNER2Extractor processing {len(split_sentences)} sentences")
        logger.info(f"Using {len(predicate_categories)} predicate categories")

        statements = []
        model = self._get_model()
        classified_texts: set[str] = set()

        for sentence in split_sentences:
            try:
                if model:
                    # Use relation extraction iterating through categories
                    # Returns ALL matches, not just the best one
                    extracted_stmts = self._extract_with_relations(sentence, model, predicate_categories)
                else:
                    # No model available - skip
                    logger.warning("No GLiNER2 model available - skipping extraction")
                    extracted_stmts = []

                for stmt in extracted_stmts:
                    statements.append(stmt)

                    # Run classifications for this statement's source text (once per unique text)
                    if model and self._classification_schemas and stmt.source_text not in classified_texts:
                        self._run_classifications(model, stmt.source_text, context)
                        classified_texts.add(stmt.source_text)

            except Exception as e:
                logger.warning(f"Error extracting from sentence: {e}")
                # No fallback - skip this sentence

        logger.info(f"GLiNER2Extractor produced {len(statements)} statements from {len(split_sentences)} sentences")
        return statements

    def _run_classifications(
        self,
        model,
        source_text: str,
        context: PipelineContext,
    ) -> None:
        """
        Run classification schemas using GLiNER2 and store results in context.

        Uses GLiNER2's create_schema() API for efficient batch classification.
        """
        if not self._classification_schemas:
            return

        # Skip if already classified this text
        if source_text in context.classification_results:
            return

        try:
            # Build schema with all classifications
            schema = model.create_schema()

            for class_schema in self._classification_schemas:
                schema = schema.classification(
                    class_schema.label_type,
                    class_schema.choices,
                )

            # Run extraction with schema
            results = model.extract(source_text, schema, include_confidence=True)

            # Store results in context
            for class_schema in self._classification_schemas:
                label_type = class_schema.label_type
                if label_type in results:
                    result_value = results[label_type]
                    # With include_confidence=True, GLiNER2 returns
                    # {'label': 'value', 'confidence': 0.95} for classifications
                    if isinstance(result_value, dict):
                        label_value = result_value.get("label", str(result_value))
                        confidence = result_value.get("confidence", 0.85)
                    else:
                        label_value = str(result_value)
                        confidence = 0.85
                    context.set_classification(
                        source_text, label_type, label_value, confidence
                    )
                    logger.debug(
                        f"GLiNER2 classified '{source_text[:50]}...' "
                        f"as {label_type}={label_value}"
                    )

        except Exception as e:
            logger.warning(f"GLiNER2 classification failed: {e}")

    def _extract_with_relations(
        self,
        sentence: SplitSentence,
        model,
        predicate_categories: dict[str, dict[str, PredicateConfig]],
    ) -> list[PipelineStatement]:
        """
        Extract using GLiNER2 relation extraction, iterating through categories.

        Iterates through each predicate category separately to stay under
        GLiNER2's ~25 label limit. Uses schema API with entities + relations.
        Returns ALL matching relations, not just the best one.

        Args:
            sentence: Split sentence from Stage 1
            model: GLiNER2 model instance
            predicate_categories: Dict of category -> predicates to use

        Returns:
            List of PipelineStatements for all relations found
        """
        logger.debug(f"Attempting relation extraction for: '{sentence.text[:80]}...'")

        # Iterate through each category separately to stay under GLiNER2's ~25 label limit
        # Use schema API with entities + relations together for better extraction
        all_relations: list[tuple[str, str, str, str, float]] = []  # (head, rel_type, tail, category, confidence)

        for category_name, category_predicates in predicate_categories.items():
            # Build relations dict with descriptions for GLiNER2 schema API
            # The .relations() method expects {relation_name: description} dict, not a list
            relations_dict = {
                pred_name: pred_config.get("description", pred_name) if isinstance(pred_config, dict) else str(pred_config)
                for pred_name, pred_config in category_predicates.items()
            }

            try:
                # Build schema with entities and relations for this category
                schema = (model.create_schema()
                    .entities(self._get_entity_types())
                    .relations(relations_dict)
                )
                result = model.extract(sentence.text, schema, include_confidence=True)

                # Get relations from this category
                relation_data = result.get("relations", result.get("relation_extraction", {}))

                # Filter to non-empty and collect relations
                for rel_type, relations in relation_data.items():
                    if not relations:
                        continue

                    for rel in relations:
                        head, tail, confidence = self._parse_relation(rel)
                        if head and tail:
                            all_relations.append((head, rel_type, tail, category_name, confidence))
                            logger.debug(f"  [{category_name}] {head} --[{rel_type}]--> {tail} (conf={confidence:.2f})")

            except Exception as e:
                logger.debug(f"  Category {category_name} extraction failed: {e}")
                continue

        total_found = len(all_relations)
        logger.debug(f"  GLiNER2 found {total_found} total relations across all categories")

        if not all_relations:
            logger.debug(f"No GLiNER2 relation match in: '{sentence.text[:60]}...'")
            return []

        # Filter by confidence threshold and sort descending
        all_relations = [(h, r, t, c, conf) for h, r, t, c, conf in all_relations if conf >= self._min_confidence]
        all_relations.sort(reverse=True, key=lambda x: x[4])  # Sort by confidence
        statements = []

        filtered_count = total_found - len(all_relations)
        if filtered_count > 0:
            logger.debug(f"  Filtered {filtered_count} relations below confidence threshold ({self._min_confidence})")

        if not all_relations:
            logger.debug(f"No relations above confidence threshold ({self._min_confidence})")
            return []

        for head, rel_type, tail, category, confidence in all_relations:
            logger.info(
                f"GLiNER2 relation match: {head} --[{rel_type}]--> {tail} "
                f"(category={category}, confidence={confidence:.2f})"
            )

            # Get entity types
            subj_type = self._infer_entity_type(head, model, sentence.text)
            obj_type = self._infer_entity_type(tail, model, sentence.text)
            logger.debug(f"  Entity types: {subj_type.value}, {obj_type.value}")

            stmt = PipelineStatement(
                subject=ExtractedEntity(
                    text=head,
                    type=subj_type,
                    confidence=confidence,
                ),
                predicate=rel_type,
                predicate_category=category,
                object=ExtractedEntity(
                    text=tail,
                    type=obj_type,
                    confidence=confidence,
                ),
                source_text=sentence.text,
                confidence_score=confidence,
                extraction_method="gliner_relation",
            )
            statements.append(stmt)

        return statements

    def _extract_with_entities(
        self,
        sentence: SplitSentence,
        model,
    ) -> Optional[PipelineStatement]:
        """
        Entity extraction mode - returns None since we don't use T5-Gemma predicates.

        This method is called when predicates are disabled. Without GLiNER2 relation
        extraction, we cannot form valid statements.
        """
        logger.debug(f"Entity extraction mode (no predicates) - skipping: '{sentence.text[:60]}...'")
        return None

    def _parse_relation(self, rel) -> tuple[str, str, float]:
        """
        Parse a relation from GLiNER2 output.

        Args:
            rel: Relation data (tuple, dict, or other format from GLiNER2)

        Returns:
            Tuple of (head_text, tail_text, confidence)
        """
        # Log the actual structure for debugging
        logger.debug(f"    Parsing relation: type={type(rel).__name__}, value={rel}")

        # Handle tuple format: (head, tail) or (head, tail, score)
        if isinstance(rel, (tuple, list)):
            if len(rel) == 2:
                head, tail = rel
                # Try to extract text if they're dicts
                head_text = head.get("text", str(head)) if isinstance(head, dict) else str(head)
                tail_text = tail.get("text", str(tail)) if isinstance(tail, dict) else str(tail)
                # Try to get confidence from dict
                head_conf = head.get("score", head.get("confidence", 0.5)) if isinstance(head, dict) else 0.5
                tail_conf = tail.get("score", tail.get("confidence", 0.5)) if isinstance(tail, dict) else 0.5
                return head_text, tail_text, min(head_conf, tail_conf)
            elif len(rel) >= 3:
                head, tail, score = rel[0], rel[1], rel[2]
                head_text = head.get("text", str(head)) if isinstance(head, dict) else str(head)
                tail_text = tail.get("text", str(tail)) if isinstance(tail, dict) else str(tail)
                return head_text, tail_text, float(score) if score else 0.5

        # Handle dict format with head/tail keys
        if isinstance(rel, dict):
            # Try different key names for head/tail
            head_data = rel.get("head") or rel.get("source") or rel.get("subject") or {}
            tail_data = rel.get("tail") or rel.get("target") or rel.get("object") or {}

            # Get overall relation confidence if available
            rel_conf = rel.get("score") or rel.get("confidence") or rel.get("prob")

            # Parse head
            if isinstance(head_data, dict):
                head = head_data.get("text") or head_data.get("name") or head_data.get("span") or ""
                head_conf = head_data.get("score") or head_data.get("confidence") or head_data.get("prob")
            else:
                head = str(head_data) if head_data else ""
                head_conf = None

            # Parse tail
            if isinstance(tail_data, dict):
                tail = tail_data.get("text") or tail_data.get("name") or tail_data.get("span") or ""
                tail_conf = tail_data.get("score") or tail_data.get("confidence") or tail_data.get("prob")
            else:
                tail = str(tail_data) if tail_data else ""
                tail_conf = None

            # Determine final confidence: prefer relation-level, then min of head/tail
            if rel_conf is not None:
                confidence = float(rel_conf)
            elif head_conf is not None and tail_conf is not None:
                confidence = min(float(head_conf), float(tail_conf))
            elif head_conf is not None:
                confidence = float(head_conf)
            elif tail_conf is not None:
                confidence = float(tail_conf)
            else:
                confidence = 0.5  # Default if no confidence found

            return head, tail, confidence

        # Unknown format
        logger.warning(f"    Unknown relation format: {type(rel).__name__}")
        return "", "", 0.0

    def _infer_entity_type(
        self,
        text: str,
        model,
        source_text: str,
    ) -> EntityType:
        """Infer entity type using GLiNER2 entity extraction."""
        try:
            result = model.extract_entities(source_text, self._get_entity_types(), include_confidence=True)
            entities = result.get("entities", {})

            text_lower = text.lower()
            for entity_type, entity_list in entities.items():
                for entity in entity_list:
                    if isinstance(entity, dict):
                        entity_text = entity.get("text", "").lower()
                    else:
                        entity_text = str(entity).lower()

                    if entity_text == text_lower or entity_text in text_lower or text_lower in entity_text:
                        return GLINER_TYPE_MAP.get(entity_type.lower(), EntityType.UNKNOWN)

        except Exception as e:
            logger.debug(f"Entity type inference failed: {e}")

        return EntityType.UNKNOWN


# Allow importing without decorator for testing
GLiNER2ExtractorClass = GLiNER2Extractor

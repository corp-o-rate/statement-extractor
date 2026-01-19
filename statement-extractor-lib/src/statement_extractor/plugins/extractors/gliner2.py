"""
GLiNER2Extractor - Stage 2 plugin that refines triples using GLiNER2.

Uses GLiNER2 for:
1. Entity extraction: Refine subject/object boundaries
2. Relation extraction: When predicate list is provided
3. Entity scoring: Score how entity-like subjects/objects are
"""

import logging
from typing import Optional

from ..base import BaseExtractorPlugin, PluginCapability
from ...pipeline.context import PipelineContext
from ...pipeline.registry import PluginRegistry
from ...models import RawTriple, PipelineStatement, ExtractedEntity, EntityType

logger = logging.getLogger(__name__)

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
    Extractor plugin that uses GLiNER2 for entity and relation refinement.

    Processes raw triples from Stage 1 and produces PipelineStatement
    objects with typed entities.
    """

    def __init__(
        self,
        predicates: Optional[list[str]] = None,
        entity_types: Optional[list[str]] = None,
    ):
        """
        Initialize the GLiNER2 extractor.

        Args:
            predicates: Optional list of predicate types for relation extraction
            entity_types: Optional list of entity types to extract (default: all)
        """
        self._predicates = predicates
        self._entity_types = entity_types or [
            "person", "organization", "company", "location",
            "city", "country", "product", "event", "date", "money", "quantity"
        ]
        self._model = None

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

    def extract(
        self,
        raw_triples: list[RawTriple],
        context: PipelineContext,
    ) -> list[PipelineStatement]:
        """
        Extract statements from raw triples using GLiNER2.

        Args:
            raw_triples: Raw triples from Stage 1
            context: Pipeline context

        Returns:
            List of PipelineStatement objects
        """
        logger.debug(f"GLiNER2Extractor processing {len(raw_triples)} triples")

        # Get options from context
        extractor_options = context.source_metadata.get("extractor_options", {})
        predicates = extractor_options.get("predicates", self._predicates)

        statements = []
        model = self._get_model()

        for raw in raw_triples:
            try:
                if model and predicates:
                    # Use relation extraction with predefined predicates
                    stmt = self._extract_with_relations(raw, model, predicates)
                elif model:
                    # Use entity extraction to refine boundaries
                    stmt = self._extract_with_entities(raw, model)
                else:
                    # Fallback: create statement from raw triple directly
                    stmt = self._create_basic_statement(raw)

                if stmt:
                    statements.append(stmt)

            except Exception as e:
                logger.warning(f"Error extracting triple: {e}")
                # Fallback to basic statement
                stmt = self._create_basic_statement(raw)
                if stmt:
                    statements.append(stmt)

        logger.info(f"GLiNER2Extractor produced {len(statements)} statements")
        return statements

    def _extract_with_relations(
        self,
        raw: RawTriple,
        model,
        predicates: list[str],
    ) -> Optional[PipelineStatement]:
        """Extract using GLiNER2 relation extraction."""
        result = model.extract_relations(raw.source_sentence, predicates)
        relation_data = result.get("relation_extraction", {})

        # Find best matching relation
        best_match = None
        best_confidence = 0.0

        for rel_type, relations in relation_data.items():
            for rel in relations:
                if isinstance(rel, tuple):
                    head, tail = rel
                    confidence = 1.0
                else:
                    head = rel.get("head", {}).get("text", "")
                    tail = rel.get("tail", {}).get("text", "")
                    confidence = min(
                        rel.get("head", {}).get("confidence", 0.5),
                        rel.get("tail", {}).get("confidence", 0.5)
                    )

                # Score based on match with raw triple
                score = confidence
                if raw.subject_text.lower() in head.lower() or head.lower() in raw.subject_text.lower():
                    score += 0.2
                if raw.object_text.lower() in tail.lower() or tail.lower() in raw.object_text.lower():
                    score += 0.2

                if score > best_confidence:
                    best_confidence = score
                    best_match = (head, rel_type, tail, confidence)

        if best_match:
            head, rel_type, tail, confidence = best_match

            # Get entity types
            subj_type = self._infer_entity_type(head, model, raw.source_sentence)
            obj_type = self._infer_entity_type(tail, model, raw.source_sentence)

            return PipelineStatement(
                subject=ExtractedEntity(
                    text=head,
                    type=subj_type,
                    confidence=confidence,
                ),
                predicate=rel_type,
                object=ExtractedEntity(
                    text=tail,
                    type=obj_type,
                    confidence=confidence,
                ),
                source_text=raw.source_sentence,
                confidence_score=confidence,
                extraction_method="gliner_relation",
            )

        # Fallback to entity extraction if no relation found
        return self._extract_with_entities(raw, model)

    def _extract_with_entities(
        self,
        raw: RawTriple,
        model,
    ) -> Optional[PipelineStatement]:
        """Extract using GLiNER2 entity extraction to refine boundaries."""
        result = model.extract_entities(raw.source_sentence, self._entity_types)
        entities = result.get("entities", {})

        # Find entities that match subject/object
        refined_subject = raw.subject_text
        refined_object = raw.object_text
        subj_type = EntityType.UNKNOWN
        obj_type = EntityType.UNKNOWN
        subj_confidence = 0.8
        obj_confidence = 0.8

        for entity_type, entity_list in entities.items():
            mapped_type = GLINER_TYPE_MAP.get(entity_type.lower(), EntityType.UNKNOWN)

            for entity in entity_list:
                if isinstance(entity, dict):
                    entity_text = entity.get("text", "")
                    confidence = entity.get("confidence", 0.8)
                else:
                    entity_text = str(entity)
                    confidence = 0.8

                entity_lower = entity_text.lower()
                subj_lower = raw.subject_text.lower()
                obj_lower = raw.object_text.lower()

                # Check subject match
                if subj_lower in entity_lower or entity_lower in subj_lower:
                    if len(entity_text) >= len(refined_subject):
                        refined_subject = entity_text
                        subj_type = mapped_type
                        subj_confidence = confidence

                # Check object match
                if obj_lower in entity_lower or entity_lower in obj_lower:
                    if len(entity_text) >= len(refined_object):
                        refined_object = entity_text
                        obj_type = mapped_type
                        obj_confidence = confidence

        # Use raw predicate (refined by later stages if needed)
        predicate = raw.predicate_text

        return PipelineStatement(
            subject=ExtractedEntity(
                text=refined_subject,
                type=subj_type,
                confidence=subj_confidence,
            ),
            predicate=predicate,
            object=ExtractedEntity(
                text=refined_object,
                type=obj_type,
                confidence=obj_confidence,
            ),
            source_text=raw.source_sentence,
            confidence_score=(subj_confidence + obj_confidence) / 2,
            extraction_method="gliner_entity",
        )

    def _create_basic_statement(self, raw: RawTriple) -> PipelineStatement:
        """Create a basic statement without GLiNER2 refinement."""
        return PipelineStatement(
            subject=ExtractedEntity(
                text=raw.subject_text,
                type=EntityType.UNKNOWN,
                confidence=raw.confidence,
            ),
            predicate=raw.predicate_text,
            object=ExtractedEntity(
                text=raw.object_text,
                type=EntityType.UNKNOWN,
                confidence=raw.confidence,
            ),
            source_text=raw.source_sentence,
            confidence_score=raw.confidence,
            extraction_method="basic",
        )

    def _infer_entity_type(
        self,
        text: str,
        model,
        source_text: str,
    ) -> EntityType:
        """Infer entity type using GLiNER2 entity extraction."""
        try:
            result = model.extract_entities(source_text, self._entity_types)
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

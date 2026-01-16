"""
GLiNER2-based triple extraction.

Uses GLiNER2 for relation extraction and entity recognition to extract
subject, predicate, and object from source text. T5-Gemma model provides
triple structure and coreference resolution, while GLiNER2 handles
linguistic analysis.

The GLiNER2 model is loaded automatically on first use.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Lazy-loaded GLiNER2 model
_model = None


def _get_model():
    """
    Lazy-load the GLiNER2 model.

    Uses the base model (205M parameters) which is CPU-optimized.
    """
    global _model
    if _model is None:
        from gliner2 import GLiNER2

        logger.info("Loading GLiNER2 model 'fastino/gliner2-base-v1'...")
        _model = GLiNER2.from_pretrained("fastino/gliner2-base-v1")
        logger.debug("GLiNER2 model loaded")
    return _model


def extract_triple_from_text(
    source_text: str,
    model_subject: str,
    model_object: str,
    model_predicate: str,
    predicates: Optional[list[str]] = None,
) -> tuple[str, str, str] | None:
    """
    Extract subject, predicate, object from source text using GLiNER2.

    Returns a GLiNER2-based triple that can be added to the candidate pool
    alongside the model's triple. The existing scoring/dedup logic will
    pick the best one.

    Args:
        source_text: The source sentence to analyze
        model_subject: Subject from T5-Gemma (used for matching and fallback)
        model_object: Object from T5-Gemma (used for matching and fallback)
        model_predicate: Predicate from T5-Gemma (used when no predicates provided)
        predicates: Optional list of predefined relation types to extract

    Returns:
        Tuple of (subject, predicate, object) from GLiNER2, or None if extraction fails
    """
    if not source_text:
        return None

    try:
        model = _get_model()

        if predicates:
            # Use relation extraction with predefined predicates
            result = model.extract_relations(source_text, predicates)

            # Find best matching relation
            relation_data = result.get("relation_extraction", {})
            best_match = None
            best_confidence = 0.0

            for rel_type, relations in relation_data.items():
                for rel in relations:
                    # Handle both tuple format and dict format
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

                    # Score based on match with model hints
                    score = confidence
                    if model_subject.lower() in head.lower() or head.lower() in model_subject.lower():
                        score += 0.2
                    if model_object.lower() in tail.lower() or tail.lower() in model_object.lower():
                        score += 0.2

                    if score > best_confidence:
                        best_confidence = score
                        best_match = (head, rel_type, tail)

            if best_match:
                logger.debug(
                    f"GLiNER2 extracted (relation): subj='{best_match[0]}', pred='{best_match[1]}', obj='{best_match[2]}'"
                )
                return best_match

        else:
            # No predicate list provided - use GLiNER2 for entity extraction
            # and extract predicate from source text using the model's hint

            # Extract entities to refine subject/object boundaries
            entity_types = [
                "person", "organization", "company", "location", "city", "country",
                "product", "event", "date", "money", "quantity"
            ]
            result = model.extract_entities(source_text, entity_types)
            entities = result.get("entities", {})

            # Find entities that match model subject/object
            refined_subject = model_subject
            refined_object = model_object

            for entity_type, entity_list in entities.items():
                for entity in entity_list:
                    entity_lower = entity.lower()
                    # Check if this entity matches or contains the model's subject/object
                    if model_subject.lower() in entity_lower or entity_lower in model_subject.lower():
                        # Use the entity text if it's more complete
                        if len(entity) >= len(refined_subject):
                            refined_subject = entity
                    if model_object.lower() in entity_lower or entity_lower in model_object.lower():
                        if len(entity) >= len(refined_object):
                            refined_object = entity

            # Use model predicate directly (T5-Gemma provides the predicate)
            if model_predicate:
                logger.debug(
                    f"GLiNER2 extracted (entity-refined): subj='{refined_subject}', pred='{model_predicate}', obj='{refined_object}'"
                )
                return (refined_subject, model_predicate, refined_object)

        return None

    except ImportError as e:
        logger.warning(f"GLiNER2 not installed: {e}")
        return None
    except Exception as e:
        logger.debug(f"GLiNER2 extraction failed: {e}")
        return None


def score_entity_content(text: str) -> float:
    """
    Score how entity-like a text is using GLiNER2 entity recognition.

    Returns:
        1.0 - Recognized as a named entity with high confidence
        0.8 - Recognized as an entity with moderate confidence
        0.6 - Partially recognized or contains entity-like content
        0.2 - Not recognized as any entity type
    """
    if not text or not text.strip():
        return 0.2

    try:
        model = _get_model()

        # Check if text is recognized as common entity types
        entity_types = [
            "person", "organization", "company", "location", "city", "country",
            "product", "event", "date", "money", "quantity"
        ]

        result = model.extract_entities(
            text,
            entity_types,
            include_confidence=True
        )

        # Result format: {'entities': {'person': [{'text': '...', 'confidence': 0.99}], ...}}
        entities_dict = result.get("entities", {})

        # Find best matching entity across all types
        best_confidence = 0.0
        text_lower = text.lower().strip()

        for entity_type, entity_list in entities_dict.items():
            for entity in entity_list:
                if isinstance(entity, dict):
                    entity_text = entity.get("text", "").lower().strip()
                    confidence = entity.get("confidence", 0.5)
                else:
                    # Fallback for string format
                    entity_text = str(entity).lower().strip()
                    confidence = 0.8

                # Check if entity covers most of the input text
                if entity_text == text_lower:
                    # Exact match
                    best_confidence = max(best_confidence, confidence)
                elif entity_text in text_lower or text_lower in entity_text:
                    # Partial match - reduce confidence
                    best_confidence = max(best_confidence, confidence * 0.8)

        if best_confidence >= 0.9:
            return 1.0
        elif best_confidence >= 0.7:
            return 0.8
        elif best_confidence >= 0.5:
            return 0.6
        elif best_confidence > 0:
            return 0.4
        else:
            return 0.2

    except Exception as e:
        logger.debug(f"Entity scoring failed for '{text}': {e}")
        return 0.5  # Neutral score on error

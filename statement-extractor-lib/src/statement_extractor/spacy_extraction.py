"""
spaCy-based triple extraction.

Uses spaCy dependency parsing to extract subject, predicate, and object
from source text. T5-Gemma model provides triple structure and coreference
resolution, while spaCy handles linguistic analysis.

The spaCy model is downloaded automatically on first use.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Lazy-loaded spaCy model
_nlp = None


def _download_model():
    """Download the spaCy model if not present."""
    import shutil
    import subprocess
    import sys

    # Direct URL to the spaCy model wheel
    MODEL_URL = "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"

    logger.info("Downloading spaCy model 'en_core_web_sm'...")

    # Try uv first (for uv-managed environments)
    uv_path = shutil.which("uv")
    if uv_path:
        try:
            result = subprocess.run(
                [uv_path, "pip", "install", MODEL_URL],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                logger.info("Successfully downloaded spaCy model via uv")
                return True
            logger.debug(f"uv pip install failed: {result.stderr}")
        except Exception as e:
            logger.debug(f"uv pip install failed: {e}")

    # Try pip directly
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", MODEL_URL],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            logger.info("Successfully downloaded spaCy model via pip")
            return True
        logger.debug(f"pip install failed: {result.stderr}")
    except Exception as e:
        logger.debug(f"pip install failed: {e}")

    # Try spacy's download as last resort
    try:
        from spacy.cli import download
        download("en_core_web_sm")
        # Check if it actually worked
        import spacy
        spacy.load("en_core_web_sm")
        logger.info("Successfully downloaded spaCy model via spacy")
        return True
    except Exception:
        pass

    logger.warning(
        "Failed to download spaCy model automatically. "
        "Please run: uv pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"
    )
    return False


def _get_nlp():
    """
    Lazy-load the spaCy model.

    Disables NER and lemmatizer for faster processing since we only
    need dependency parsing. Automatically downloads the model if not present.
    """
    global _nlp
    if _nlp is None:
        import spacy

        # Try to load the model, download if not present
        try:
            _nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
            logger.debug("Loaded spaCy model for extraction")
        except OSError:
            # Model not found, try to download it
            if _download_model():
                _nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
                logger.debug("Loaded spaCy model after download")
            else:
                raise OSError(
                    "spaCy model not found and automatic download failed. "
                    "Please run: python -m spacy download en_core_web_sm"
                )
    return _nlp


def _get_full_noun_phrase(token) -> str:
    """
    Get the full noun phrase for a token, including compounds and modifiers.
    """
    # Get all tokens in the subtree that form the noun phrase
    phrase_tokens = []

    # Collect compound modifiers and the token itself
    for t in token.subtree:
        # Include compounds, adjectives, determiners, and the head noun
        if t.dep_ in ("compound", "amod", "det", "poss", "nummod", "nmod") or t == token:
            phrase_tokens.append(t)

    # Sort by position and join
    phrase_tokens.sort(key=lambda x: x.i)
    return " ".join([t.text for t in phrase_tokens])


def _extract_verb_phrase(verb_token) -> str:
    """
    Extract the full verb phrase including auxiliaries and particles.
    """
    parts = []

    # Collect auxiliaries that come before the verb
    for child in verb_token.children:
        if child.dep_ in ("aux", "auxpass") and child.i < verb_token.i:
            parts.append((child.i, child.text))

    # Add the main verb
    parts.append((verb_token.i, verb_token.text))

    # Collect particles and prepositions that are part of phrasal verbs
    for child in verb_token.children:
        if child.dep_ == "prt" and child.i > verb_token.i:
            parts.append((child.i, child.text))
        # Include prepositions for phrasal verbs like "announced by"
        elif child.dep_ == "agent" and child.i > verb_token.i:
            # For passive constructions, include "by"
            parts.append((child.i, child.text))

    # Sort by position and join
    parts.sort(key=lambda x: x[0])
    return " ".join([p[1] for p in parts])


def _match_entity_boundaries(
    spacy_text: str,
    model_text: str,
    source_text: str,
) -> str:
    """
    Match entity boundaries between spaCy extraction and model hint.

    If model text is a superset that includes spaCy text, use model text
    for better entity boundaries (e.g., "Apple" -> "Apple Inc.").
    """
    spacy_lower = spacy_text.lower()
    model_lower = model_text.lower()

    # If model text contains spaCy text, prefer model text
    if spacy_lower in model_lower:
        return model_text

    # If spaCy text contains model text, prefer spaCy text
    if model_lower in spacy_lower:
        return spacy_text

    # If they overlap significantly, prefer the one that appears in source
    if spacy_text in source_text:
        return spacy_text
    if model_text in source_text:
        return model_text

    # Default to spaCy extraction
    return spacy_text


def _extract_spacy_triple(doc, model_subject: str, model_object: str, source_text: str) -> tuple[str | None, str | None, str | None]:
    """Extract subject, predicate, object from spaCy doc."""
    # Find the root verb
    root = None
    for token in doc:
        if token.dep_ == "ROOT":
            root = token
            break

    if root is None:
        return None, None, None

    # Extract predicate from root verb
    predicate = None
    if root.pos_ == "VERB":
        predicate = _extract_verb_phrase(root)
    elif root.pos_ == "AUX":
        predicate = root.text

    # Extract subject (nsubj, nsubjpass)
    subject = None
    for child in root.children:
        if child.dep_ in ("nsubj", "nsubjpass"):
            subject = _get_full_noun_phrase(child)
            break

    # If no direct subject, check parent
    if subject is None and root.head != root:
        for child in root.head.children:
            if child.dep_ in ("nsubj", "nsubjpass"):
                subject = _get_full_noun_phrase(child)
                break

    # Extract object (dobj, pobj, attr, oprd)
    obj = None
    for child in root.children:
        if child.dep_ in ("dobj", "attr", "oprd"):
            obj = _get_full_noun_phrase(child)
            break
        elif child.dep_ == "prep":
            for pchild in child.children:
                if pchild.dep_ == "pobj":
                    obj = _get_full_noun_phrase(pchild)
                    break
            if obj:
                break
        elif child.dep_ == "agent":
            for pchild in child.children:
                if pchild.dep_ == "pobj":
                    obj = _get_full_noun_phrase(pchild)
                    break
            if obj:
                break

    # Match against model values for better entity boundaries
    if subject:
        subject = _match_entity_boundaries(subject, model_subject, source_text)
    if obj:
        obj = _match_entity_boundaries(obj, model_object, source_text)

    return subject, predicate, obj


def extract_triple_from_text(
    source_text: str,
    model_subject: str,
    model_object: str,
    model_predicate: str,
) -> tuple[str, str, str] | None:
    """
    Extract subject, predicate, object from source text using spaCy.

    Returns a spaCy-based triple that can be added to the candidate pool
    alongside the model's triple. The existing scoring/dedup logic will
    pick the best one.

    Args:
        source_text: The source sentence to analyze
        model_subject: Subject from T5-Gemma (used for entity boundary matching)
        model_object: Object from T5-Gemma (used for entity boundary matching)
        model_predicate: Predicate from T5-Gemma (unused, kept for API compat)

    Returns:
        Tuple of (subject, predicate, object) from spaCy, or None if extraction fails
    """
    if not source_text:
        return None

    try:
        nlp = _get_nlp()
        doc = nlp(source_text)
        spacy_subject, spacy_predicate, spacy_object = _extract_spacy_triple(
            doc, model_subject, model_object, source_text
        )

        # Only return if we got at least a predicate
        if spacy_predicate:
            logger.debug(
                f"spaCy extracted: subj='{spacy_subject}', pred='{spacy_predicate}', obj='{spacy_object}'"
            )
            return (
                spacy_subject or model_subject,
                spacy_predicate,
                spacy_object or model_object,
            )

        return None

    except OSError as e:
        logger.debug(f"Cannot load spaCy model: {e}")
        return None
    except Exception as e:
        logger.debug(f"spaCy extraction failed: {e}")
        return None


def extract_triple_by_predicate_split(
    source_text: str,
    predicate: str,
) -> tuple[str, str, str] | None:
    """
    Extract subject and object by splitting the source text around the predicate.

    This is useful when the predicate is known but subject/object boundaries
    are uncertain. Uses the predicate as an anchor point.

    Args:
        source_text: The source sentence
        predicate: The predicate (verb phrase) to split on

    Returns:
        Tuple of (subject, predicate, object) or None if split fails
    """
    if not source_text or not predicate:
        return None

    # Find the predicate in the source text (case-insensitive)
    source_lower = source_text.lower()
    pred_lower = predicate.lower()

    pred_pos = source_lower.find(pred_lower)
    if pred_pos < 0:
        # Try finding just the main verb (first word of predicate)
        main_verb = pred_lower.split()[0] if pred_lower.split() else ""
        if main_verb and len(main_verb) > 2:
            pred_pos = source_lower.find(main_verb)
            if pred_pos >= 0:
                # Adjust to use the actual predicate length for splitting
                predicate = main_verb

    if pred_pos < 0:
        return None

    # Extract subject (text before predicate, trimmed)
    subject = source_text[:pred_pos].strip()

    # Extract object (text after predicate, trimmed)
    pred_end = pred_pos + len(predicate)
    obj = source_text[pred_end:].strip()

    # Clean up: remove trailing punctuation from object
    obj = obj.rstrip('.,;:!?')

    # Clean up: remove leading articles/prepositions from object if very short
    obj_words = obj.split()
    if obj_words and obj_words[0].lower() in ('a', 'an', 'the', 'to', 'of', 'for'):
        if len(obj_words) > 1:
            obj = ' '.join(obj_words[1:])

    # Validate: both subject and object should have meaningful content
    if len(subject) < 2 or len(obj) < 2:
        return None

    logger.debug(
        f"Predicate-split extracted: subj='{subject}', pred='{predicate}', obj='{obj}'"
    )

    return (subject, predicate, obj)


# Keep old function for backwards compatibility
def infer_predicate(
    subject: str,
    obj: str,
    source_text: str,
) -> Optional[str]:
    """
    Infer the predicate from source text using dependency parsing.

    DEPRECATED: Use extract_triple_from_text instead.
    """
    result = extract_triple_from_text(
        source_text=source_text,
        model_subject=subject,
        model_object=obj,
        model_predicate="",
    )
    if result:
        _, predicate, _ = result
        return predicate if predicate else None
    return None

"""
RunPod Serverless Handler for Statement Extractor

This handler uses the corp-extractor library to extract structured statements from text.
Uses Diverse Beam Search (https://arxiv.org/abs/1610.02424) for high-quality extraction.

v0.2.0: Returns JSON with confidence scores, canonical predicates, and full extraction metadata.
"""

import hashlib
import json
import logging
import os
from collections import OrderedDict

import runpod
from statement_extractor import StatementExtractor, ExtractionOptions, ScoringConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model configuration
MODEL_ID = os.environ.get("MODEL_ID", "Corp-o-Rate-Community/statement-extractor")

# Cache configuration (default 10GB)
MAX_CACHE_SIZE_BYTES = int(os.environ.get("MAX_CACHE_SIZE_BYTES", 10 * 1024 * 1024 * 1024))

# Extraction options from environment
NUM_BEAMS = int(os.environ.get("NUM_RETURN_SEQUENCES", 4))
MIN_STATEMENT_RATIO = float(os.environ.get("MIN_STATEMENT_RATIO", 1.0))
MAX_EXTRACTION_ATTEMPTS = int(os.environ.get("MAX_EXTRACTION_ATTEMPTS", 3))

# Output format: "json" (default, v0.2.0+) or "xml" (legacy)
OUTPUT_FORMAT = os.environ.get("OUTPUT_FORMAT", "json")

# Global extractor instance (loaded once, reused across requests)
extractor: StatementExtractor | None = None

# In-memory LRU cache for results (persists while worker is warm)
result_cache: OrderedDict[str, str] = OrderedDict()
cache_size_bytes = 0


def get_cache_key(text: str) -> str:
    """Generate a cache key from input text."""
    return hashlib.sha256(text.encode()).hexdigest()


def get_entry_size(key: str, value: str) -> int:
    """Estimate memory size of a cache entry in bytes."""
    return len(key.encode()) + len(value.encode())


def evict_if_needed(new_entry_size: int):
    """Evict oldest entries if cache would exceed size limit."""
    global cache_size_bytes

    while result_cache and (cache_size_bytes + new_entry_size) > MAX_CACHE_SIZE_BYTES:
        oldest_key, oldest_value = result_cache.popitem(last=False)
        evicted_size = get_entry_size(oldest_key, oldest_value)
        cache_size_bytes -= evicted_size
        logger.info(f"Evicted cache entry: {oldest_key[:16]}... (freed {evicted_size} bytes)")


def load_extractor():
    """Load the extractor if not already loaded."""
    global extractor

    if extractor is not None:
        logger.info("Extractor already loaded")
        return

    logger.info(f"Loading extractor with model: {MODEL_ID}")
    extractor = StatementExtractor(model_id=MODEL_ID)
    logger.info(f"Extractor loaded on device: {extractor.device}")


def extract_statements(text: str, output_format: str = "json") -> str:
    """Extract statements from text with caching.

    Args:
        text: Input text (with or without <page> tags)
        output_format: "json" (v0.2.0+, includes confidence) or "xml" (legacy)

    Returns:
        JSON string with full extraction result, or XML string for legacy mode
    """
    global extractor, cache_size_bytes

    # Include format in cache key so json/xml cached separately
    cache_key = get_cache_key(f"{output_format}:{text}")
    if cache_key in result_cache:
        result_cache.move_to_end(cache_key)
        logger.info(f"Cache hit for key: {cache_key[:16]}...")
        return result_cache[cache_key]

    # Ensure extractor is loaded
    load_extractor()

    # Configure extraction options with scoring for confidence scores
    options = ExtractionOptions(
        num_beams=NUM_BEAMS,
        min_statement_ratio=MIN_STATEMENT_RATIO,
        max_attempts=MAX_EXTRACTION_ATTEMPTS,
        merge_beams=True,
        embedding_dedup=True,
        scoring_config=ScoringConfig(),  # Enable quality scoring
    )

    # Extract using the library
    if output_format == "xml":
        # Legacy XML output (no confidence scores)
        result = extractor.extract_as_xml(text, options)
    else:
        # New JSON output with full metadata (v0.2.0+)
        result = extractor.extract_as_json(text, options, indent=None)

    logger.info(f"Extracted {len(result)} chars ({output_format})")

    # Store in cache
    entry_size = get_entry_size(cache_key, result)
    evict_if_needed(entry_size)
    result_cache[cache_key] = result
    cache_size_bytes += entry_size
    logger.info(f"Cached result for key: {cache_key[:16]}... (entries: {len(result_cache)}, size: {cache_size_bytes / 1024 / 1024:.1f}MB)")

    return result


def handler(job):
    """
    RunPod serverless handler function.

    Expected input format:
    {
        "input": {
            "text": "<page>Your text here</page>",
            "format": "json"  // optional: "json" (default) or "xml"
        }
    }

    Returns (JSON format, v0.2.0+):
    {
        "output": {
            "statements": [...],  // Array of statement objects with confidence
            "source_text": "..."
        },
        "format": "json"
    }

    Returns (XML format, legacy):
    {
        "output": "<statements>...</statements>",
        "format": "xml"
    }
    """
    logger.info(f"Received job: {job}")

    job_input = job.get("input", {})
    logger.info(f"Job input: {job_input}")

    # Get text from input - support both "text" and "prompt" keys
    text = job_input.get("text") or job_input.get("prompt", "")

    if not text:
        logger.error(f"No text provided. Job keys: {list(job.keys())}, Input keys: {list(job_input.keys()) if isinstance(job_input, dict) else 'not a dict'}")
        return {"error": "No text provided. Send {\"input\": {\"text\": \"your text here\"}}"}

    # Get output format (default to JSON for v0.2.0+)
    output_format = job_input.get("format", OUTPUT_FORMAT).lower()
    if output_format not in ("json", "xml"):
        output_format = "json"

    # Wrap in page tags if not already wrapped
    if not text.startswith("<page>"):
        text = f"<page>{text}</page>"

    logger.info(f"Processing text of length: {len(text)}, format: {output_format}")

    try:
        result = extract_statements(text, output_format)
        logger.info(f"Extraction complete, output length: {len(result)}")

        # For JSON format, parse the string back to dict for cleaner response
        if output_format == "json":
            return {"output": json.loads(result), "format": "json"}
        else:
            return {"output": result, "format": "xml"}
    except Exception as e:
        logger.exception("Error during extraction")
        return {"error": str(e)}


# Load extractor at startup for faster cold starts
load_extractor()

# Start the serverless handler
runpod.serverless.start({"handler": handler})

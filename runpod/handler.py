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
from statement_extractor import StatementExtractor, ExtractionOptions, ScoringConfig, PredicateTaxonomy, PredicateComparisonConfig

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

# Canonical predicates for corporate/ESG domain
# These are normalized forms that extracted predicates can be mapped to
CANONICAL_PREDICATES = [
    # Ownership & Corporate Structure
    "owned by", "parent organization", "ultimate parent", "has subsidiary",
    "acquired", "merged with", "spun off from", "controlling shareholder",
    "minority shareholder", "beneficial owner", "person of significant control",
    "nominee for", "succeeded by", "preceded by",
    # Investment & Finance
    "investor", "funded by", "creditor", "debtor", "vc backed by", "pe backed by",
    "ipo underwriter", "bond holder",
    # Leadership & Employment
    "chief executive officer", "chief financial officer", "chief operating officer",
    "founder", "board member", "chairperson", "company secretary", "employee",
    "former employee", "former director", "advisor", "consultant",
    # Organizational Structure
    "division of", "department of",
    # Supply Chain
    "supplier", "customer", "manufacturer", "distributor", "contractor",
    "outsources to", "subcontractor", "raw material source",
    # Geography & Jurisdiction
    "headquarters", "located in", "operates in", "facility in", "registered in",
    "tax residence", "offshore entity in", "branch in", "citizenship", "formed in", "residence",
    # Legal & Regulatory
    "sued", "sued by", "fined by", "regulated by", "licensed by", "sanctioned by",
    "investigated by", "settled with", "consent decree with", "debarred by",
    # Political
    "lobbies", "donated to", "endorsed by", "member of", "sponsored by",
    "lobbied by", "pac contribution", "revolving door",
    # Environmental & Social
    "polluted", "affected community", "displaced", "deforested", "benefited",
    "restored", "employed in", "invested in community", "violated rights",
    "emitted ghg", "water usage", "waste disposal",
    # Products & IP
    "brand of", "product of", "trademark of", "licensed from", "white label for",
    "recalls", "developer", "publisher",
    # Business Relationships
    "partner", "joint venture with", "franchisee of", "distributor for",
    "licensed to", "exclusive dealer", "operator",
    # Personal Relationships
    "spouse", "relative", "associate", "co-founder", "classmate", "club member",
    # Classification
    "industry", "competitor", "similar to", "same sector", "peer of", "instance of",
    # Events & Mentions
    "mentioned with", "accused of", "praised for", "criticized for",
    "announced", "rumored", "participant",
]

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


def extract_statements(
    text: str,
    output_format: str = "json",
    use_canonical_predicates: bool = False,
    similarity_threshold: float = 0.5,
) -> str:
    """Extract statements from text with caching.

    Args:
        text: Input text (with or without <page> tags)
        output_format: "json" (v0.2.0+, includes confidence) or "xml" (legacy)
        use_canonical_predicates: If True, normalize predicates to canonical forms
        similarity_threshold: Threshold for predicate similarity matching (default 0.5)

    Returns:
        JSON string with full extraction result, or XML string for legacy mode
    """
    global extractor, cache_size_bytes

    # Include format, taxonomy option, and threshold in cache key
    cache_key = get_cache_key(f"{output_format}:canonical={use_canonical_predicates}:thresh={similarity_threshold}:{text}")
    if cache_key in result_cache:
        result_cache.move_to_end(cache_key)
        logger.info(f"Cache hit for key: {cache_key[:16]}...")
        return result_cache[cache_key]

    # Ensure extractor is loaded
    load_extractor()

    # Configure predicate comparison with custom threshold (only for taxonomy, not dedup)
    predicate_config = PredicateComparisonConfig(
        similarity_threshold=similarity_threshold,  # For taxonomy matching
        # dedup_threshold uses library default (0.65)
    )

    # Configure extraction options with scoring for confidence scores
    options = ExtractionOptions(
        num_beams=NUM_BEAMS,
        min_statement_ratio=MIN_STATEMENT_RATIO,
        max_attempts=MAX_EXTRACTION_ATTEMPTS,
        merge_beams=True,
        embedding_dedup=True,
        scoring_config=ScoringConfig(),  # Enable quality scoring
        predicate_config=predicate_config,
    )

    # Add predicate taxonomy if requested
    if use_canonical_predicates:
        options.predicate_taxonomy = PredicateTaxonomy(predicates=CANONICAL_PREDICATES)
        logger.info(f"Using canonical predicate taxonomy with {len(CANONICAL_PREDICATES)} predicates, threshold={similarity_threshold}")

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

    # Get canonical predicates option
    use_canonical_predicates = job_input.get("useCanonicalPredicates", False)

    # Get similarity threshold (default 0.5 for broader matching)
    similarity_threshold = float(job_input.get("similarityThreshold", 0.5))
    # Clamp to valid range
    similarity_threshold = max(0.0, min(1.0, similarity_threshold))

    # Wrap in page tags if not already wrapped
    if not text.startswith("<page>"):
        text = f"<page>{text}</page>"

    logger.info(f"Processing text of length: {len(text)}, format: {output_format}, canonical: {use_canonical_predicates}, threshold: {similarity_threshold}")

    try:
        result = extract_statements(text, output_format, use_canonical_predicates, similarity_threshold)
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

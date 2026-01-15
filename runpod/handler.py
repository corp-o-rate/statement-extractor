"""
RunPod Serverless Handler for Statement Extractor

This handler uses the corp-extractor library to extract structured statements from text.
Uses Diverse Beam Search (https://arxiv.org/abs/1610.02424) for high-quality extraction.

v0.3.0: Async handler with TRUE batched inference for efficient GPU utilization.
        Multiple requests are tokenized together and processed in a single forward pass.
"""

import asyncio
import hashlib
import json
import logging
import os
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional

import runpod
import torch
from statement_extractor import (
    ExtractionOptions,
    PredicateComparisonConfig,
    PredicateTaxonomy,
    ScoringConfig,
    StatementExtractor,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model configuration
MODEL_ID = os.environ.get("MODEL_ID", "Corp-o-Rate-Community/statement-extractor")

# Cache configuration (default 10GB)
MAX_CACHE_SIZE_BYTES = int(os.environ.get("MAX_CACHE_SIZE_BYTES", 10 * 1024 * 1024 * 1024))

# Extraction options from environment
DEFAULT_NUM_BEAMS = int(os.environ.get("NUM_RETURN_SEQUENCES", 8))
MIN_STATEMENT_RATIO = float(os.environ.get("MIN_STATEMENT_RATIO", 1.0))
MAX_EXTRACTION_ATTEMPTS = int(os.environ.get("MAX_EXTRACTION_ATTEMPTS", 3))
DEFAULT_MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", 4096))

# Output format: "json" (default, v0.2.0+) or "xml" (legacy)
OUTPUT_FORMAT = os.environ.get("OUTPUT_FORMAT", "json")

# Concurrency and batching settings
MAX_CONCURRENCY = int(os.environ.get("MAX_CONCURRENCY", 16))
MAX_BATCH_SIZE = int(os.environ.get("MAX_BATCH_SIZE", 8))
BATCH_TIMEOUT_MS = int(os.environ.get("BATCH_TIMEOUT_MS", 50))  # Short timeout for batching

# VRAM-based batch size scaling
AUTO_SCALE_BATCH = os.environ.get("AUTO_SCALE_BATCH", "true").lower() == "true"
# Approximate VRAM per item in batch (includes KV cache, activations)
VRAM_PER_BATCH_ITEM_GB = float(os.environ.get("VRAM_PER_BATCH_ITEM_GB", 1.5))
BASE_VRAM_GB = float(os.environ.get("BASE_VRAM_GB", 2.5))

# Canonical predicates for corporate/ESG domain
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


@dataclass
class BatchRequest:
    """A request waiting to be batched."""
    text: str
    original_text: str  # Without page tags, for post-processing
    output_format: str
    use_canonical_predicates: bool
    similarity_threshold: float
    future: asyncio.Future
    cache_key: str
    timestamp: float


# Global state
extractor: Optional[StatementExtractor] = None
result_cache: OrderedDict[str, str] = OrderedDict()
cache_size_bytes = 0

# Batching state (initialized lazily)
batch_queue: Optional[asyncio.Queue] = None
batch_processor_task: Optional[asyncio.Task] = None
inference_lock: Optional[asyncio.Lock] = None
_async_initialized = False

# Calculated max batch size based on VRAM
calculated_max_batch: int = MAX_BATCH_SIZE


def get_total_vram_gb() -> float:
    """Get total GPU VRAM in GB."""
    if not torch.cuda.is_available():
        return 0.0
    try:
        return torch.cuda.get_device_properties(0).total_memory / (1024**3)
    except Exception as e:
        logger.warning(f"Failed to get VRAM info: {e}")
        return 0.0


def calculate_max_batch_size() -> int:
    """Calculate maximum batch size based on available VRAM."""
    if not AUTO_SCALE_BATCH:
        return MAX_BATCH_SIZE

    total_vram = get_total_vram_gb()
    if total_vram == 0:
        return MAX_BATCH_SIZE

    # Available VRAM for batching = total - base model
    available = total_vram - BASE_VRAM_GB
    max_by_vram = int(available / VRAM_PER_BATCH_ITEM_GB)

    # Clamp to configured max
    result = max(1, min(max_by_vram, MAX_BATCH_SIZE))

    logger.info(f"VRAM: {total_vram:.1f}GB total, {available:.1f}GB available")
    logger.info(f"Calculated max batch size: {result}")

    return result


def get_cache_key(text: str, output_format: str, use_canonical: bool, threshold: float) -> str:
    """Generate a cache key from input parameters."""
    key_str = f"{output_format}:canonical={use_canonical}:thresh={threshold}:{text}"
    return hashlib.sha256(key_str.encode()).hexdigest()


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


def cache_result(cache_key: str, result: str):
    """Store result in cache."""
    global cache_size_bytes
    entry_size = get_entry_size(cache_key, result)
    evict_if_needed(entry_size)
    result_cache[cache_key] = result
    cache_size_bytes += entry_size


def load_extractor():
    """Load the extractor if not already loaded."""
    global extractor, calculated_max_batch

    if extractor is not None:
        return

    logger.info(f"Loading extractor with model: {MODEL_ID}")
    extractor = StatementExtractor(model_id=MODEL_ID)
    logger.info(f"Extractor loaded on device: {extractor.device}")

    # Calculate max batch size after model is loaded
    calculated_max_batch = calculate_max_batch_size()


def batched_generate(texts: list[str]) -> list[str]:
    """
    Run batched generation on multiple texts.

    This is the key function that utilizes GPU VRAM efficiently by
    processing multiple inputs in a single forward pass.
    """
    global extractor

    load_extractor()

    model = extractor.model
    tokenizer = extractor.tokenizer
    device = extractor.device

    # Tokenize all texts together with padding
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        max_length=4096,
        truncation=True,
        padding=True,  # Pad to longest in batch
    ).to(device)

    logger.info(f"Batched tokenization: {len(texts)} texts, max_len={inputs['input_ids'].shape[1]}")

    # Generate for entire batch at once
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
            max_length=None,  # Override model default, use max_new_tokens only
            num_beams=DEFAULT_NUM_BEAMS,
            num_beam_groups=DEFAULT_NUM_BEAMS,
            num_return_sequences=1,  # One best result per input
            diversity_penalty=1.0,
            do_sample=False,
            top_p=None,  # Override model config to suppress warning
            top_k=None,  # Override model config to suppress warning
            trust_remote_code=True,
            custom_generate="transformers-community/group-beam-search",
        )

    # Decode each output
    results = []
    end_tag = "</statements>"

    for output in outputs:
        decoded = tokenizer.decode(output, skip_special_tokens=True)

        # Truncate at </statements>
        if end_tag in decoded:
            end_pos = decoded.find(end_tag) + len(end_tag)
            decoded = decoded[:end_pos]

        results.append(decoded)

    logger.info(f"Batched generation complete: {len(results)} outputs")
    return results


def post_process_result(
    xml_output: str,
    original_text: str,
    output_format: str,
    use_canonical_predicates: bool,
    similarity_threshold: float,
) -> str:
    """
    Post-process raw XML output into final result format.

    Applies deduplication, predicate normalization, and scoring.
    """
    global extractor

    # For XML format, just deduplicate and return
    if output_format == "xml":
        # Use extractor's deduplication
        from statement_extractor.canonicalization import deduplicate_statements_exact
        import xml.etree.ElementTree as ET

        try:
            root = ET.fromstring(xml_output)
            if root.tag == 'statements':
                seen = set()
                unique = []
                for stmt in root.findall('stmt'):
                    key = (
                        stmt.findtext('subject', '').strip().lower(),
                        stmt.findtext('predicate', '').strip().lower(),
                        stmt.findtext('object', '').strip().lower(),
                    )
                    if key not in seen:
                        seen.add(key)
                        unique.append(stmt)
                new_root = ET.Element('statements')
                for stmt in unique:
                    new_root.append(stmt)
                return ET.tostring(new_root, encoding='unicode')
        except ET.ParseError:
            pass
        return xml_output

    # For JSON format, use full extraction pipeline for quality scoring
    predicate_config = PredicateComparisonConfig(
        similarity_threshold=similarity_threshold,
    )

    options = ExtractionOptions(
        num_beams=DEFAULT_NUM_BEAMS,
        min_statement_ratio=MIN_STATEMENT_RATIO,
        max_attempts=1,  # Already generated, just post-process
        max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
        merge_beams=True,
        embedding_dedup=True,
        scoring_config=ScoringConfig(),
        predicate_config=predicate_config,
    )

    if use_canonical_predicates:
        options.predicate_taxonomy = PredicateTaxonomy(predicates=CANONICAL_PREDICATES)

    # Parse XML to statements and apply full post-processing
    statements = extractor._parse_xml_to_statements(xml_output)

    if statements and options.embedding_dedup:
        try:
            comparer = extractor._get_predicate_comparer(options)
            if comparer:
                statements = comparer.deduplicate_statements(statements)
                if options.predicate_taxonomy:
                    statements = comparer.normalize_predicates(statements)
        except Exception as e:
            logger.warning(f"Embedding dedup failed: {e}")

    # Build result
    from statement_extractor.models import ExtractionResult
    result = ExtractionResult(statements=statements, source_text=original_text)
    return result.model_dump_json(indent=None)


def process_batch_sync(requests: list[BatchRequest]):
    """
    Process a batch of requests with true batched inference.

    This function:
    1. Collects all texts
    2. Runs batched model.generate()
    3. Post-processes each result individually
    """
    if not requests:
        return

    batch_size = len(requests)
    logger.info(f"Processing TRUE BATCH of {batch_size} requests")

    start_time = time.time()

    try:
        # Collect texts for batched generation
        texts = [req.text for req in requests]

        # Run batched generation (single GPU forward pass for all inputs)
        raw_outputs = batched_generate(texts)

        gen_time = time.time() - start_time
        logger.info(f"Batched generation took {gen_time:.2f}s for {batch_size} items ({gen_time/batch_size:.2f}s/item)")

        # Post-process each result individually
        for req, raw_output in zip(requests, raw_outputs):
            try:
                result = post_process_result(
                    xml_output=raw_output,
                    original_text=req.original_text,
                    output_format=req.output_format,
                    use_canonical_predicates=req.use_canonical_predicates,
                    similarity_threshold=req.similarity_threshold,
                )

                # Cache the result
                cache_result(req.cache_key, result)
                req.future.set_result(result)

            except Exception as e:
                logger.exception(f"Error post-processing request: {e}")
                req.future.set_exception(e)

    except Exception as e:
        logger.exception(f"Error in batched generation: {e}")
        for req in requests:
            if not req.future.done():
                req.future.set_exception(e)

    total_time = time.time() - start_time
    logger.info(f"Total batch processing: {total_time:.2f}s for {batch_size} items")


async def process_batch(requests: list[BatchRequest]):
    """Async wrapper for batch processing."""
    global inference_lock

    if not requests:
        return

    async with inference_lock:
        # Run in thread pool to avoid blocking event loop
        await asyncio.get_event_loop().run_in_executor(
            None,
            process_batch_sync,
            requests,
        )


async def batch_processor():
    """
    Background task that collects requests and processes them in batches.

    Waits up to BATCH_TIMEOUT_MS to collect multiple requests before processing.
    """
    global batch_queue, calculated_max_batch

    logger.info(f"Batch processor started (max_batch={calculated_max_batch}, timeout={BATCH_TIMEOUT_MS}ms)")

    while True:
        batch: list[BatchRequest] = []

        try:
            # Wait for first request
            first_request = await batch_queue.get()
            batch.append(first_request)

            # Try to collect more requests within timeout
            deadline = time.time() + (BATCH_TIMEOUT_MS / 1000)
            max_batch = calculated_max_batch

            while len(batch) < max_batch:
                remaining = deadline - time.time()
                if remaining <= 0:
                    break

                try:
                    request = await asyncio.wait_for(
                        batch_queue.get(),
                        timeout=remaining
                    )
                    batch.append(request)
                except asyncio.TimeoutError:
                    break

            # Process the batch
            logger.info(f"Collected batch of {len(batch)} requests (max={max_batch})")
            await process_batch(batch)

        except Exception as e:
            logger.exception(f"Batch processor error: {e}")
            for req in batch:
                if not req.future.done():
                    req.future.set_exception(e)


async def submit_request(
    text: str,
    original_text: str,
    output_format: str,
    use_canonical_predicates: bool,
    similarity_threshold: float,
) -> str:
    """Submit a request to the batching queue and wait for result."""
    global batch_queue

    # Check cache first
    cache_key = get_cache_key(text, output_format, use_canonical_predicates, similarity_threshold)
    if cache_key in result_cache:
        result_cache.move_to_end(cache_key)
        logger.info(f"Cache hit: {cache_key[:16]}...")
        return result_cache[cache_key]

    loop = asyncio.get_event_loop()
    future = loop.create_future()

    request = BatchRequest(
        text=text,
        original_text=original_text,
        output_format=output_format,
        use_canonical_predicates=use_canonical_predicates,
        similarity_threshold=similarity_threshold,
        future=future,
        cache_key=cache_key,
        timestamp=time.time(),
    )

    await batch_queue.put(request)
    return await future


async def handler(job):
    """
    Async RunPod serverless handler function.

    Expected input format:
    {
        "input": {
            "text": "<page>Your text here</page>",
            "format": "json"  // optional: "json" (default) or "xml"
        }
    }
    """
    # Initialize async components on first request
    await ensure_async_initialized()

    logger.info(f"Received job: {job.get('id', 'unknown')}")

    job_input = job.get("input", {})

    # Get text from input
    text = job_input.get("text") or job_input.get("prompt", "")

    if not text:
        logger.error(f"No text provided")
        return {"error": "No text provided. Send {\"input\": {\"text\": \"your text here\"}}"}

    # Store original text before adding tags
    original_text = text

    # Get output format
    output_format = job_input.get("format", OUTPUT_FORMAT).lower()
    if output_format not in ("json", "xml"):
        output_format = "json"

    # Get options
    use_canonical_predicates = job_input.get("useCanonicalPredicates", False)
    similarity_threshold = float(job_input.get("similarityThreshold", 0.5))
    similarity_threshold = max(0.0, min(1.0, similarity_threshold))

    # Wrap in page tags if not already wrapped
    if not text.startswith("<page>"):
        text = f"<page>{text}</page>"

    logger.info(f"Queuing text of length: {len(text)}, format: {output_format}")

    try:
        result = await submit_request(
            text=text,
            original_text=original_text,
            output_format=output_format,
            use_canonical_predicates=use_canonical_predicates,
            similarity_threshold=similarity_threshold,
        )

        logger.info(f"Extraction complete, output length: {len(result)}")

        if output_format == "json":
            return {"output": json.loads(result), "format": "json"}
        else:
            return {"output": result, "format": "xml"}

    except Exception as e:
        logger.exception("Error during extraction")
        return {"error": str(e)}


def concurrency_modifier(current_concurrency: int) -> int:
    """Return maximum concurrent jobs this worker can handle."""
    return MAX_CONCURRENCY


async def ensure_async_initialized():
    """Lazily initialize async components on first request."""
    global batch_queue, batch_processor_task, inference_lock, _async_initialized

    if _async_initialized:
        return

    batch_queue = asyncio.Queue()
    inference_lock = asyncio.Lock()
    batch_processor_task = asyncio.create_task(batch_processor())
    _async_initialized = True
    logger.info("Async components initialized")


# Load extractor at startup for faster cold starts
load_extractor()

# Start the serverless handler with concurrency support
runpod.serverless.start({
    "handler": handler,
    "concurrency_modifier": concurrency_modifier,
})

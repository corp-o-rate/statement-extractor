"""
RunPod Serverless Handler for Statement Extractor

This handler loads the T5-Gemma statement extraction model and processes
requests to extract structured statements from text.
"""

import runpod
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import logging
import hashlib
import os
import re
from collections import OrderedDict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model configuration
MODEL_ID = "Corp-o-Rate-Community/statement-extractor"

# Global model and tokenizer (loaded once, reused across requests)
model = None
tokenizer = None

# Cache configuration (default 10GB)
MAX_CACHE_SIZE_BYTES = int(os.environ.get("MAX_CACHE_SIZE_BYTES", 10 * 1024 * 1024 * 1024))

# Number of candidate sequences to generate (default 4)
NUM_RETURN_SEQUENCES = int(os.environ.get("NUM_RETURN_SEQUENCES", 4))

# Retry configuration for under-extraction
# If statements < sentences * MIN_STATEMENT_RATIO, retry extraction
MIN_STATEMENT_RATIO = float(os.environ.get("MIN_STATEMENT_RATIO", 0.5))
MAX_EXTRACTION_ATTEMPTS = int(os.environ.get("MAX_EXTRACTION_ATTEMPTS", 3))

# In-memory LRU cache for results (persists while worker is warm)
result_cache: OrderedDict[str, str] = OrderedDict()
cache_size_bytes = 0


def get_cache_key(text: str) -> str:
    """Generate a cache key from input text."""
    return hashlib.sha256(text.encode()).hexdigest()


def count_sentences(text: str) -> int:
    """Count approximate number of sentences in text."""
    # Remove XML tags
    clean_text = re.sub(r'<[^>]+>', '', text)
    # Split on sentence-ending punctuation
    sentences = re.split(r'[.!?]+', clean_text)
    # Filter out empty strings and very short fragments
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    return max(1, len(sentences))


def count_statements(xml_output: str) -> int:
    """Count number of <stmt> tags in the output."""
    return len(re.findall(r'<stmt>', xml_output))


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


def load_model():
    """Load the model and tokenizer into memory."""
    global model, tokenizer

    if model is not None:
        logger.info("Model already loaded")
        return

    logger.info(f"Loading model: {MODEL_ID}")

    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
    )

    # Load model with appropriate dtype
    if device == "cuda":
        model = AutoModelForSeq2SeqLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
        )
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
        )
        model = model.to(device)

    logger.info("Model loaded successfully")


def run_single_extraction(inputs, device) -> str:
    """Run a single extraction attempt and return the best candidate."""
    global model, tokenizer

    # Generate multiple diverse candidate outputs using diverse beam search
    num_seqs = NUM_RETURN_SEQUENCES
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=2048,
            num_beams=num_seqs,
            num_beam_groups=num_seqs,
            num_return_sequences=num_seqs,
            diversity_penalty=1.0,
            do_sample=False,
        )

    # Decode all sequences and select the longest valid one
    end_tag = "</statements>"
    candidates = []
    for i, output in enumerate(outputs):
        decoded = tokenizer.decode(output, skip_special_tokens=True)

        # Truncate at </statements>
        if end_tag in decoded:
            end_pos = decoded.find(end_tag) + len(end_tag)
            decoded = decoded[:end_pos]
            candidates.append(decoded)

    # Select longest candidate
    if candidates:
        return max(candidates, key=len)
    else:
        # Fallback to first output if none have valid closing tag
        return tokenizer.decode(outputs[0], skip_special_tokens=True)


def extract_statements(text: str) -> str:
    """Extract statements from the input text with retry logic."""
    global model, tokenizer, cache_size_bytes

    # Check cache first
    cache_key = get_cache_key(text)
    if cache_key in result_cache:
        result_cache.move_to_end(cache_key)
        logger.info(f"Cache hit for key: {cache_key[:16]}...")
        return result_cache[cache_key]

    # Ensure model is loaded
    load_model()

    device = next(model.parameters()).device

    # Tokenize input
    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=4096,
        truncation=True,
    ).to(device)

    # Count sentences for quality check
    num_sentences = count_sentences(text)
    min_expected_statements = int(num_sentences * MIN_STATEMENT_RATIO)
    logger.info(f"Input has ~{num_sentences} sentences, expecting at least {min_expected_statements} statements")

    # Run extraction with retry logic
    all_results = []
    for attempt in range(MAX_EXTRACTION_ATTEMPTS):
        result = run_single_extraction(inputs, device)
        num_stmts = count_statements(result)
        all_results.append((result, num_stmts))
        logger.info(f"Attempt {attempt + 1}/{MAX_EXTRACTION_ATTEMPTS}: {num_stmts} statements, {len(result)} chars")

        # If we have enough statements, stop retrying
        if num_stmts >= min_expected_statements:
            logger.info(f"Got {num_stmts} statements (>= {min_expected_statements}), accepting result")
            break
        elif attempt < MAX_EXTRACTION_ATTEMPTS - 1:
            logger.info(f"Only {num_stmts} statements (< {min_expected_statements}), retrying...")

    # Select the best result (longest, which typically has most statements)
    best_result = max(all_results, key=lambda x: len(x[0]))[0]
    best_stmt_count = count_statements(best_result)
    logger.info(f"Selected best result: {best_stmt_count} statements, {len(best_result)} chars from {len(all_results)} attempts")

    # Store in cache
    entry_size = get_entry_size(cache_key, best_result)
    evict_if_needed(entry_size)
    result_cache[cache_key] = best_result
    cache_size_bytes += entry_size
    logger.info(f"Cached result for key: {cache_key[:16]}... (entries: {len(result_cache)}, size: {cache_size_bytes / 1024 / 1024:.1f}MB)")

    return best_result


def handler(job):
    """
    RunPod serverless handler function.

    Expected input format:
    {
        "input": {
            "text": "<page>Your text here</page>"
        }
    }

    Returns:
    {
        "output": "<statements>...</statements>"
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

    # Wrap in page tags if not already wrapped
    if not text.startswith("<page>"):
        text = f"<page>{text}</page>"

    logger.info(f"Processing text of length: {len(text)}")

    try:
        result = extract_statements(text)
        logger.info(f"Extraction complete, output length: {len(result)}")
        return {"output": result}
    except Exception as e:
        logger.exception("Error during extraction")
        return {"error": str(e)}


# Load model at startup for faster cold starts
load_model()

# Start the serverless handler
runpod.serverless.start({"handler": handler})

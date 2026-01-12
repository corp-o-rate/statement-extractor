"""
RunPod Serverless Handler for Statement Extractor

This handler loads the T5-Gemma statement extraction model and processes
requests to extract structured statements from text.
"""

import runpod
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
import logging
import hashlib
import os
from collections import OrderedDict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model configuration
MODEL_ID = "Corp-o-Rate-Community/statement-extractor"

# Global model and tokenizer (loaded once, reused across requests)
model = None
tokenizer = None
stop_token_ids = None

# Cache configuration (default 10GB)
MAX_CACHE_SIZE_BYTES = int(os.environ.get("MAX_CACHE_SIZE_BYTES", 10 * 1024 * 1024 * 1024))

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


class StopOnToken(StoppingCriteria):
    """Stop generation when a specific token sequence is generated."""

    def __init__(self, stop_ids: list):
        self.stop_ids = stop_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # Check if any of the stop token IDs appear in the last generated tokens
        for stop_id in self.stop_ids:
            if input_ids[0, -1].item() == stop_id:
                return True
        return False


def load_model():
    """Load the model and tokenizer into memory."""
    global model, tokenizer, stop_token_ids

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

    # Get the token ID for </statements> to use as stopping criteria
    stop_token_ids = tokenizer.encode("</statements>", add_special_tokens=False)
    logger.info(f"Stop token IDs for </statements>: {stop_token_ids}")

    logger.info("Model loaded successfully")


def extract_statements(text: str) -> str:
    """Extract statements from the input text."""
    global model, tokenizer, stop_token_ids, cache_size_bytes

    # Check cache first
    cache_key = get_cache_key(text)
    if cache_key in result_cache:
        # Move to end for LRU ordering
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

    # Create stopping criteria to stop at </statements>
    stopping_criteria = StoppingCriteriaList([StopOnToken(stop_token_ids)])

    # Generate output
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=2048,
            num_beams=4,
            do_sample=False,
            stopping_criteria=stopping_criteria,
        )

    # Decode and return
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Safety: also truncate at first </statements> in case stopping didn't work perfectly
    end_tag = "</statements>"
    if end_tag in result:
        end_pos = result.find(end_tag) + len(end_tag)
        result = result[:end_pos]

    # Store in cache with size tracking
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

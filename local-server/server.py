"""
Local API server for Statement Extractor model.

This server loads the T5-Gemma 2 model locally and provides an API endpoint
for statement extraction without rate limits.

Features (matching RunPod handler):
- Diverse beam search with multiple candidates
- Retry logic for under-extraction
- In-memory LRU cache

Usage:
    # Using .env file
    uv run python server.py

    # Using CLI arguments (overrides .env)
    uv run python server.py --model-path ../model --port 8000

Environment variables (.env):
    MODEL_PATH: Path to the model directory
    PORT: Port to run the server on (default: 8000)
    HOST: Host to bind to (default: 0.0.0.0)
    NUM_RETURN_SEQUENCES: Number of candidate sequences (default: 4)
    MIN_STATEMENT_RATIO: Minimum statements per sentence ratio (default: 1.0)
    MAX_EXTRACTION_ATTEMPTS: Max retry attempts (default: 3)
    MAX_CACHE_SIZE_BYTES: In-memory cache size limit (default: 1GB)
"""

import argparse
import hashlib
import logging
import re
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import torch
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Load .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Server configuration from environment variables or .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    model_path: Path = Field(
        default=Path.home() / "corp_var" / "models" / "page_splitter",
        description="Path to the model directory",
    )
    port: int = Field(
        default=8000,
        description="Port to run the server on",
    )
    host: str = Field(
        default="0.0.0.0",
        description="Host to bind to",
    )
    log_level: str = Field(
        default="INFO",
        description="Logging level",
    )
    num_return_sequences: int = Field(
        default=4,
        description="Number of candidate sequences to generate",
    )
    min_statement_ratio: float = Field(
        default=1.0,
        description="Minimum statements per sentence ratio before retry",
    )
    max_extraction_attempts: int = Field(
        default=3,
        description="Maximum extraction retry attempts",
    )
    max_cache_size_bytes: int = Field(
        default=1 * 1024 * 1024 * 1024,  # 1GB default for local
        description="Maximum in-memory cache size in bytes",
    )


# Global settings instance
settings = Settings()

app = FastAPI(
    title="Statement Extractor API",
    description="Local API for T5-Gemma 2 statement extraction model",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model and tokenizer
model: Optional[AutoModelForSeq2SeqLM] = None
tokenizer: Optional[AutoTokenizer] = None
device: str = "cpu"

# In-memory LRU cache for results
result_cache: OrderedDict[str, str] = OrderedDict()
cache_size_bytes: int = 0


class ExtractRequest(BaseModel):
    text: str


class ExtractResponse(BaseModel):
    output: str
    success: bool
    cached: bool = False
    error: Optional[str] = None


def get_cache_key(text: str) -> str:
    """Generate a cache key from input text."""
    return hashlib.sha256(text.encode()).hexdigest()


def get_entry_size(key: str, value: str) -> int:
    """Estimate memory size of a cache entry in bytes."""
    return len(key.encode()) + len(value.encode())


def evict_if_needed(new_entry_size: int) -> None:
    """Evict oldest entries if cache would exceed size limit."""
    global cache_size_bytes

    while result_cache and (cache_size_bytes + new_entry_size) > settings.max_cache_size_bytes:
        oldest_key, oldest_value = result_cache.popitem(last=False)
        evicted_size = get_entry_size(oldest_key, oldest_value)
        cache_size_bytes -= evicted_size
        logger.info(f"Evicted cache entry: {oldest_key[:16]}... (freed {evicted_size} bytes)")


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


def load_model(model_path: Path) -> None:
    """Load the model and tokenizer."""
    global model, tokenizer, device

    logger.info(f"Loading model from {model_path}...")

    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    # Determine device
    if torch.cuda.is_available():
        device = "cuda"
        logger.info("Using CUDA GPU")
    elif torch.backends.mps.is_available():
        device = "mps"
        logger.info("Using Apple MPS")
    else:
        device = "cpu"
        logger.info("Using CPU (this will be slow)")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)

    # Load model
    model = AutoModelForSeq2SeqLM.from_pretrained(
        str(model_path),
        torch_dtype=torch.bfloat16 if device != "cpu" else torch.float32,
        trust_remote_code=True,
    )
    model = model.to(device)
    model.eval()

    logger.info("Model loaded successfully!")


def run_single_extraction(inputs) -> str:
    """Run a single extraction attempt and return the best candidate."""
    global model, tokenizer

    num_seqs = settings.num_return_sequences

    # Generate multiple diverse candidate outputs using diverse beam search
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=2048,
            num_beams=num_seqs,
            num_beam_groups=num_seqs,
            num_return_sequences=num_seqs,
            diversity_penalty=1.0,
            do_sample=False,
            trust_remote_code=True,
        )

    # Decode all sequences and select the longest valid one
    end_tag = "</statements>"
    candidates = []
    for output in outputs:
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


def extract_with_retry(text: str) -> tuple[str, bool]:
    """
    Extract statements from text with retry logic.
    Returns (result, cached) tuple.
    """
    global model, tokenizer, cache_size_bytes

    # Check cache first
    cache_key = get_cache_key(text)
    if cache_key in result_cache:
        result_cache.move_to_end(cache_key)
        logger.info(f"Cache hit for key: {cache_key[:16]}...")
        return result_cache[cache_key], True

    # Tokenize input
    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=4096,
        truncation=True,
    ).to(device)

    # Count sentences for quality check
    num_sentences = count_sentences(text)
    min_expected_statements = int(num_sentences * settings.min_statement_ratio)
    logger.info(f"Input has ~{num_sentences} sentences, expecting at least {min_expected_statements} statements")

    # Run extraction with retry logic
    all_results = []
    for attempt in range(settings.max_extraction_attempts):
        result = run_single_extraction(inputs)
        num_stmts = count_statements(result)
        all_results.append((result, num_stmts))
        logger.info(f"Attempt {attempt + 1}/{settings.max_extraction_attempts}: {num_stmts} statements, {len(result)} chars")

        # If we have enough statements, stop retrying
        if num_stmts >= min_expected_statements:
            logger.info(f"Got {num_stmts} statements (>= {min_expected_statements}), accepting result")
            break
        elif attempt < settings.max_extraction_attempts - 1:
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

    return best_result, False


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "device": device,
        "model_path": str(settings.model_path),
        "cache_entries": len(result_cache),
        "cache_size_mb": round(cache_size_bytes / 1024 / 1024, 2),
    }


@app.post("/extract", response_model=ExtractResponse)
async def extract_statements(request: ExtractRequest):
    """Extract statements from text."""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        text = request.text.strip()
        if not text:
            raise HTTPException(status_code=400, detail="Empty text")

        # Ensure text is wrapped in page tags
        if not text.startswith("<page>"):
            text = f"<page>{text}</page>"

        logger.info(f"Processing text of length {len(text)}")

        result, cached = extract_with_retry(text)

        logger.info(f"Generated output of length {len(result)} (cached: {cached})")

        return ExtractResponse(output=result, success=True, cached=cached)

    except Exception as e:
        logger.error(f"Error during extraction: {e}")
        return ExtractResponse(output="", success=False, error=str(e))


@app.delete("/cache")
async def clear_cache():
    """Clear the in-memory cache."""
    global result_cache, cache_size_bytes
    count = len(result_cache)
    result_cache = OrderedDict()
    cache_size_bytes = 0
    logger.info(f"Cleared {count} cache entries")
    return {"cleared": count}


def main():
    """Main entry point."""
    import uvicorn

    parser = argparse.ArgumentParser(
        description="Statement Extractor API Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Environment variables (can also be set in .env file):
  MODEL_PATH              Path to the model directory
  PORT                    Port to run the server on (default: 8000)
  HOST                    Host to bind to (default: 0.0.0.0)
  LOG_LEVEL               Logging level (default: INFO)
  NUM_RETURN_SEQUENCES    Candidates to generate (default: 4)
  MIN_STATEMENT_RATIO     Min statements/sentence ratio (default: 1.0)
  MAX_EXTRACTION_ATTEMPTS Max retry attempts (default: 3)
  MAX_CACHE_SIZE_BYTES    Cache size limit (default: 1GB)

Example .env file:
  MODEL_PATH=/path/to/model
  PORT=8000
  HOST=0.0.0.0
        """,
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help=f"Path to the model directory (default: {settings.model_path})",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help=f"Port to run the server on (default: {settings.port})",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help=f"Host to bind to (default: {settings.host})",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="Path to .env file (default: .env in current directory)",
    )

    args = parser.parse_args()

    # Reload settings if custom env file specified
    if args.env_file:
        load_dotenv(args.env_file, override=True)
        global settings
        settings = Settings()

    # CLI arguments override environment/settings
    model_path = args.model_path or settings.model_path
    port = args.port or settings.port
    host = args.host or settings.host

    # Set log level
    logging.getLogger().setLevel(settings.log_level.upper())

    logger.info("Configuration:")
    logger.info(f"  Model path: {model_path}")
    logger.info(f"  Host: {host}")
    logger.info(f"  Port: {port}")
    logger.info(f"  Num return sequences: {settings.num_return_sequences}")
    logger.info(f"  Min statement ratio: {settings.min_statement_ratio}")
    logger.info(f"  Max extraction attempts: {settings.max_extraction_attempts}")
    logger.info(f"  Max cache size: {settings.max_cache_size_bytes / 1024 / 1024:.0f}MB")

    # Load model
    load_model(model_path)

    # Run server
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()

"""
Local API server for Statement Extractor model.

This server loads the T5-Gemma 2 model locally and provides an API endpoint
for statement extraction without rate limits.

Usage:
    # Using .env file
    uv run python server.py

    # Using CLI arguments (overrides .env)
    uv run python server.py --model-path ../model --port 8000

Environment variables (.env):
    MODEL_PATH: Path to the model directory
    PORT: Port to run the server on (default: 8000)
    HOST: Host to bind to (default: 0.0.0.0)
"""

import argparse
import logging
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


class ExtractRequest(BaseModel):
    text: str


class ExtractResponse(BaseModel):
    output: str
    success: bool
    error: Optional[str] = None


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


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "device": device,
        "model_path": str(settings.model_path),
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

        # Tokenize
        inputs = tokenizer(
            text,
            return_tensors="pt",
            max_length=4096,
            truncation=True,
        ).to(device)

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=2048,
                num_beams=4,
                do_sample=False,
            )

        # Decode
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)

        logger.info(f"Generated output of length {len(result)}")

        return ExtractResponse(output=result, success=True)

    except Exception as e:
        logger.error(f"Error during extraction: {e}")
        return ExtractResponse(output="", success=False, error=str(e))


def main():
    """Main entry point."""
    import uvicorn

    parser = argparse.ArgumentParser(
        description="Statement Extractor API Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Environment variables (can also be set in .env file):
  MODEL_PATH  Path to the model directory
  PORT        Port to run the server on (default: 8000)
  HOST        Host to bind to (default: 0.0.0.0)
  LOG_LEVEL   Logging level (default: INFO)

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

    logger.info(f"Configuration:")
    logger.info(f"  Model path: {model_path}")
    logger.info(f"  Host: {host}")
    logger.info(f"  Port: {port}")

    # Load model
    load_model(model_path)

    # Run server
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()

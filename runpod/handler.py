"""
RunPod Serverless Handler for Statement Extractor

This handler loads the T5-Gemma statement extraction model and processes
requests to extract structured statements from text.
"""

import runpod
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model configuration
MODEL_ID = "Corp-o-Rate-Community/statement-extractor"

# Global model and tokenizer (loaded once, reused across requests)
model = None
tokenizer = None


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


def extract_statements(text: str) -> str:
    """Extract statements from the input text."""
    global model, tokenizer

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

    # Generate output
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=2048,
            num_beams=4,
            do_sample=False,
        )

    # Decode and return
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
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

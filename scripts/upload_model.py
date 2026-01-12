#!/usr/bin/env python3
"""
Upload the Statement Extractor model to HuggingFace.

Usage:
    # Using .env file
    uv run python upload_model.py

    # Using environment variables
    HF_TOKEN=your_token uv run python upload_model.py

    # With custom model path
    uv run python upload_model.py --model-path /path/to/model

Environment variables (.env):
    HF_TOKEN: HuggingFace API token (required)
    MODEL_PATH: Path to the model directory (optional)
    REPO_ID: HuggingFace repository ID (optional)
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Default configuration
DEFAULT_MODEL_PATH = Path.home() / "corp_var" / "models" / "page_splitter"
DEFAULT_REPO_ID = "Corp-o-Rate-Community/statement-extractor"

MODEL_CARD_TEMPLATE = '''---
license: mit
language:
- en
tags:
- statement-extraction
- named-entity-recognition
- t5
- gemma
- seq2seq
- nlp
- information-extraction
- corp-o-rate
pipeline_tag: text2text-generation
---

# Statement Extractor (T5-Gemma 2)

A fine-tuned T5-Gemma 2 model for extracting structured statements from text. Part of [corp-o-rate.com](https://corp-o-rate.com).

## Model Description

This model extracts subject-predicate-object triples from unstructured text, with automatic entity type recognition and coreference resolution.

- **Architecture**: T5-Gemma 2 (270M-270M, 540M total parameters)
- **Training Data**: 77,515 examples from corporate and news documents
- **Final Eval Loss**: 0.209
- **Max Input Length**: 4,096 tokens
- **Max Output Length**: 2,048 tokens

## Usage

### Python

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

model = AutoModelForSeq2SeqLM.from_pretrained(
    "{repo_id}",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(
    "{repo_id}",
    trust_remote_code=True,
)

text = "Apple Inc. announced a commitment to carbon neutrality by 2030."
inputs = tokenizer(f"<page>{{text}}</page>", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=2048, num_beams=4)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)
```

### Input Format

Wrap your text in `<page>` tags:

```
<page>Your text here...</page>
```

### Output Format

The model outputs XML with extracted statements:

```xml
<statements>
  <stmt>
    <subject type="ORG">Apple Inc.</subject>
    <object type="EVENT">carbon neutrality by 2030</object>
    <predicate>committed to</predicate>
    <text>Apple Inc. committed to achieving carbon neutrality by 2030.</text>
  </stmt>
</statements>
```

## Entity Types

| Type | Description |
|------|-------------|
| ORG | Organizations (companies, agencies) |
| PERSON | People (names, titles) |
| GPE | Geopolitical entities (countries, cities) |
| LOC | Locations (mountains, rivers) |
| PRODUCT | Products (devices, services) |
| EVENT | Events (announcements, meetings) |
| WORK_OF_ART | Creative works (reports, books) |
| LAW | Legal documents |
| DATE | Dates and time periods |
| MONEY | Monetary values |
| PERCENT | Percentages |
| QUANTITY | Quantities and measurements |

## Demo

Try the interactive demo at [statement-extractor.vercel.app](https://statement-extractor.vercel.app)

## Training

- Base model: `google/t5gemma-2-270m-270m`
- Training examples: 77,515
- Final eval loss: 0.209
- Training with refinement phase (LR=1e-6, epochs=0.2)
- Beam search: num_beams=4

## About corp-o-rate

This model is part of [corp-o-rate.com](https://corp-o-rate.com) - an AI-powered platform for ESG analysis and corporate accountability.

## License

MIT License
'''


def main():
    from huggingface_hub import HfApi, login

    parser = argparse.ArgumentParser(
        description="Upload Statement Extractor model to HuggingFace",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Environment variables (can also be set in .env file):
  HF_TOKEN    HuggingFace API token (required)
  MODEL_PATH  Path to the model directory
  REPO_ID     HuggingFace repository ID

Example .env file:
  HF_TOKEN=hf_xxxxxxxxxxxx
  MODEL_PATH=/path/to/model
  REPO_ID=username/model-name
        """,
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help=f"Path to the model directory (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help=f"HuggingFace repository ID (default: {DEFAULT_REPO_ID})",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="Path to .env file (default: .env in current directory)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be uploaded without actually uploading",
    )

    args = parser.parse_args()

    # Reload .env if custom file specified
    if args.env_file:
        load_dotenv(args.env_file, override=True)

    # Get configuration from args or environment
    token = os.environ.get("HF_TOKEN")
    model_path = args.model_path or Path(os.environ.get("MODEL_PATH", str(DEFAULT_MODEL_PATH)))
    repo_id = args.repo_id or os.environ.get("REPO_ID", DEFAULT_REPO_ID)

    # Validate token
    if not token:
        print("Error: HF_TOKEN not set")
        print("Set it in .env file or as environment variable:")
        print("  export HF_TOKEN=your_token_here")
        print("  # or")
        print("  echo 'HF_TOKEN=your_token_here' >> .env")
        sys.exit(1)

    # Validate model path
    if not model_path.exists():
        print(f"Error: Model path does not exist: {model_path}")
        sys.exit(1)

    print(f"Configuration:")
    print(f"  Model path: {model_path}")
    print(f"  Repository: {repo_id}")
    print()

    # List files to upload
    files_to_upload = [
        "model.safetensors",
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "training_metadata.json",
    ]

    existing_files = []
    missing_files = []
    for filename in files_to_upload:
        filepath = model_path / filename
        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024 * 1024)
            existing_files.append((filename, size_mb))
        else:
            missing_files.append(filename)

    print("Files to upload:")
    for filename, size_mb in existing_files:
        print(f"  {filename} ({size_mb:.1f} MB)")

    if missing_files:
        print("\nMissing files (will be skipped):")
        for filename in missing_files:
            print(f"  {filename}")

    if args.dry_run:
        print("\n[DRY RUN] Would upload the above files to:")
        print(f"  https://huggingface.co/{repo_id}")
        return

    print()
    input("Press Enter to continue with upload (Ctrl+C to cancel)...")

    # Login
    print("\nLogging in to HuggingFace...")
    login(token=token)

    # Initialize API
    api = HfApi()

    # Create repository if it doesn't exist
    try:
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
        print(f"Repository created/verified: {repo_id}")
    except Exception as e:
        print(f"Error creating repository: {e}")
        sys.exit(1)

    # Generate model card with correct repo ID
    model_card = MODEL_CARD_TEMPLATE.format(repo_id=repo_id)

    # Upload model card
    print("\nUploading model card...")
    api.upload_file(
        path_or_fileobj=model_card.encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
    )

    # Upload model files
    print("\nUploading model files (this may take a while for large files)...")
    for filename, size_mb in existing_files:
        filepath = model_path / filename
        print(f"  Uploading {filename} ({size_mb:.1f} MB)...")
        api.upload_file(
            path_or_fileobj=str(filepath),
            path_in_repo=filename,
            repo_id=repo_id,
            repo_type="model",
        )

    print(f"\nUpload complete!")
    print(f"Model available at: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
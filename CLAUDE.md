# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Statement Extractor is a web demo for the T5-Gemma 2 statement extraction model. It transforms unstructured text into structured subject-predicate-object triples with entity type recognition.

## Commands

### Frontend (Next.js)
```bash
pnpm install     # Install dependencies
pnpm dev         # Start dev server at localhost:3000
pnpm build       # Production build
pnpm lint        # Run ESLint
```

### Local Model Server
```bash
cd local-server
uv sync                        # Install Python dependencies
uv run python server.py        # Start FastAPI server at localhost:8000
```

### RunPod Deployment
```bash
cd runpod
# Build for Linux/amd64 (required on Mac)
docker build --platform linux/amd64 -t statement-extractor-runpod .
```

### Upload Model to HuggingFace
```bash
cd scripts
uv sync
uv run python upload_model.py
```

### Python Library (statement-extractor)
```bash
cd statement-extractor-lib
uv sync                        # Install dependencies
uv run pytest                  # Run tests
uv build                       # Build package
uv publish                     # Publish to PyPI (requires credentials)
```

## Architecture

### Three Deployment Modes
The frontend can connect to the model via three backends (configured by environment variables):
1. **RunPod Serverless** (`RUNPOD_ENDPOINT_ID`, `RUNPOD_API_KEY`) - Production, pay-per-use GPU
2. **Local Server** (`LOCAL_MODEL_URL`) - Self-hosted FastAPI server
3. **HuggingFace Inference** - Fallback, rate-limited

### Directory Structure
- `src/` - Next.js frontend (React 19, Tailwind CSS, D3.js for graph visualization)
- `src/app/api/extract/` - API route that proxies to model backends
- `local-server/` - FastAPI server for local model inference (uv-managed)
- `runpod/` - Docker + handler for RunPod serverless deployment
- `scripts/` - HuggingFace upload utilities (uv-managed)
- `statement-extractor-lib/` - Python library for statement extraction (PyPI package)

### Model I/O Format
- **Input**: Text wrapped in `<page>` tags
- **Output**: XML with `<statements>` containing `<stmt>` elements with `<subject>`, `<object>`, `<predicate>`, `<text>`
- Entity types: ORG, PERSON, GPE, LOC, PRODUCT, EVENT, WORK_OF_ART, LAW, DATE, MONEY, PERCENT, QUANTITY

### Key Technical Notes
- Uses [Diverse Beam Search](https://arxiv.org/abs/1610.02424) (Vijayakumar et al., 2016) for high-quality extraction
- T5Gemma2 requires `transformers` dev version from GitHub (not PyPI)
- RunPod requires `--platform linux/amd64` when building Docker on Mac
- Model uses bfloat16 on GPU, float32 on CPU
- Generation stops at `</statements>` tag to prevent runaway output
- **v0.4.0**: Uses GLiNER2 (205M params) for entity recognition and relation extraction instead of spaCy
- GLiNER2 is CPU-optimized and handles NER, relation extraction, and structured data extraction

### GLiNER2 Integration (v0.4.0)
The library uses GLiNER2 for:
1. **Entity extraction**: Refines subject/object boundaries from T5-Gemma output
2. **Relation extraction**: When `predicates` list is provided, uses GLiNER2's relation extraction
3. **Entity scoring**: Scores how "entity-like" subjects/objects are (replaces spaCy POS tagging)

Two extraction modes:
- **With predicates list**: Uses `extract_relations()` for predefined relation types
- **Without predicates**: Uses entity extraction to refine boundaries + predicate split for verb extraction

### Python Library API
```python
from statement_extractor import extract_statements, extract_statements_as_json

# Returns Pydantic models
result = extract_statements("Apple announced a new iPhone.")
for stmt in result:
    print(f"{stmt.subject.text} -> {stmt.predicate} -> {stmt.object.text}")

# With predefined predicates (GLiNER2 relation extraction)
from statement_extractor import ExtractionOptions
options = ExtractionOptions(predicates=["works_for", "founded", "acquired"])
result = extract_statements("John works for Apple Inc.", options)

# Other formats
json_str = extract_statements_as_json("...")
xml_str = extract_statements_as_xml("...")
dict_data = extract_statements_as_dict("...")
```

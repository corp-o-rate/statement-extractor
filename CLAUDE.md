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

### Model I/O Format
- **Input**: Text wrapped in `<page>` tags
- **Output**: XML with `<statements>` containing `<stmt>` elements with `<subject>`, `<object>`, `<predicate>`, `<text>`
- Entity types: ORG, PERSON, GPE, LOC, PRODUCT, EVENT, WORK_OF_ART, LAW, DATE, MONEY, PERCENT, QUANTITY

### Key Technical Notes
- T5Gemma2 requires `transformers` dev version from GitHub (not PyPI)
- RunPod requires `--platform linux/amd64` when building Docker on Mac
- Model uses bfloat16 on GPU, float32 on CPU
- Generation stops at `</statements>` tag to prevent runaway output

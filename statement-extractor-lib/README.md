# Corp Extractor

Extract structured subject-predicate-object statements from unstructured text using the T5-Gemma 2 model.

[![PyPI version](https://img.shields.io/pypi/v/corp-extractor.svg)](https://pypi.org/project/corp-extractor/)
[![Python 3.10+](https://img.shields.io/pypi/pyversions/corp-extractor.svg)](https://pypi.org/project/corp-extractor/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- **Structured Extraction**: Converts unstructured text into subject-predicate-object triples
- **Entity Type Recognition**: Identifies 12 entity types (ORG, PERSON, GPE, LOC, PRODUCT, EVENT, etc.)
- **GLiNER2 Integration** *(v0.4.0)*: Uses GLiNER2 (205M params) for entity recognition and relation extraction
- **Predefined Predicates** *(v0.4.0)*: Optional `--predicates` list for GLiNER2 relation extraction mode
- **Entity-based Scoring** *(v0.4.0)*: Confidence combines semantic similarity (50%) + entity recognition scores (25% each)
- **Multi-Candidate Extraction**: Generates 2 candidates per statement (hybrid, GLiNER2-only)
- **Best Triple Selection**: Keeps only highest-scoring triple per source (use `--all-triples` to keep all)
- **Extraction Method Tracking**: Each statement includes `extraction_method` field (hybrid, gliner, model)
- **Beam Merging**: Combines top beams for better coverage instead of picking one
- **Embedding-based Dedup**: Uses semantic similarity to detect near-duplicate predicates
- **Predicate Taxonomies**: Normalize predicates to canonical forms via embeddings
- **Contextualized Matching**: Compares full "Subject Predicate Object" against source text for better accuracy
- **Entity Type Merging**: Automatically merges UNKNOWN entity types with specific types during deduplication
- **Reversal Detection**: Detects and corrects subject-object reversals using embedding comparison
- **Command Line Interface**: Full-featured CLI for terminal usage
- **Multiple Output Formats**: Get results as Pydantic models, JSON, XML, or dictionaries

## Installation

```bash
pip install corp-extractor
```

The GLiNER2 model (205M params) is downloaded automatically on first use.

**Note**: This package requires `transformers>=5.0.0` for T5-Gemma2 model support.

**For GPU support**, install PyTorch with CUDA first:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install corp-extractor
```

**For Apple Silicon (M1/M2/M3)**, MPS acceleration is automatically detected:
```bash
pip install corp-extractor  # MPS used automatically
```

## Quick Start

```python
from statement_extractor import extract_statements

result = extract_statements("""
    Apple Inc. announced the iPhone 15 at their September event.
    Tim Cook presented the new features to customers worldwide.
""")

for stmt in result:
    print(f"{stmt.subject.text} ({stmt.subject.type})")
    print(f"  --[{stmt.predicate}]--> {stmt.object.text}")
    print(f"  Confidence: {stmt.confidence_score:.2f}")  # NEW in v0.2.0
```

## Command Line Interface

The library includes a CLI for quick extraction from the terminal.

### Install Globally (Recommended)

For best results, install globally first:

```bash
# Using uv (recommended)
uv tool install "corp-extractor[embeddings]"

# Using pipx
pipx install "corp-extractor[embeddings]"

# Using pip
pip install "corp-extractor[embeddings]"

# Then use anywhere
corp-extractor "Your text here"
```

### Quick Run with uvx

Run directly without installing using [uv](https://docs.astral.sh/uv/):

```bash
uvx corp-extractor "Apple announced a new iPhone."
```

**Note**: First run downloads the model (~1.5GB) which may take a few minutes.

### Usage Examples

```bash
# Extract from text argument
corp-extractor "Apple Inc. announced the iPhone 15 at their September event."

# Extract from file
corp-extractor -f article.txt

# Pipe from stdin
cat article.txt | corp-extractor -

# Output as JSON
corp-extractor "Tim Cook is CEO of Apple." --json

# Output as XML
corp-extractor -f article.txt --xml

# Verbose output with confidence scores
corp-extractor -f article.txt --verbose

# Use more beams for better quality
corp-extractor -f article.txt --beams 8

# Use custom predicate taxonomy
corp-extractor -f article.txt --taxonomy predicates.txt

# Use GPU explicitly
corp-extractor -f article.txt --device cuda
```

### CLI Options

```
Usage: corp-extractor [OPTIONS] [TEXT]

Options:
  -f, --file PATH              Read input from file
  -o, --output [table|json|xml] Output format (default: table)
  --json                       Output as JSON (shortcut)
  --xml                        Output as XML (shortcut)
  -b, --beams INTEGER          Number of beams (default: 4)
  --diversity FLOAT            Diversity penalty (default: 1.0)
  --max-tokens INTEGER         Max tokens to generate (default: 2048)
  --no-dedup                   Disable deduplication
  --no-embeddings              Disable embedding-based dedup (faster)
  --no-merge                   Disable beam merging
  --no-gliner                  Disable GLiNER2 extraction (use raw model output)
  --predicates TEXT            Comma-separated predicate types for GLiNER2 relation extraction
  --all-triples                Keep all candidate triples (default: best per source)
  --dedup-threshold FLOAT      Deduplication threshold (default: 0.65)
  --min-confidence FLOAT       Min confidence filter (default: 0)
  --taxonomy PATH              Load predicate taxonomy from file
  --taxonomy-threshold FLOAT   Taxonomy matching threshold (default: 0.5)
  --device [auto|cuda|mps|cpu] Device to use (default: auto)
  -v, --verbose                Show confidence scores and metadata
  -q, --quiet                  Suppress progress messages
  --version                    Show version
  --help                       Show this message
```

## New in v0.2.0: Quality Scoring & Beam Merging

By default, the library now:
- **Scores each triple** for groundedness based on whether entities appear in source text
- **Merges top beams** instead of selecting one, improving coverage
- **Uses embeddings** to detect semantically similar predicates ("bought" ≈ "acquired")

```python
from statement_extractor import ExtractionOptions, ScoringConfig

# Precision mode - filter low-confidence triples
scoring = ScoringConfig(min_confidence=0.7)
options = ExtractionOptions(scoring_config=scoring)
result = extract_statements(text, options)

# Access confidence scores
for stmt in result:
    print(f"{stmt} (confidence: {stmt.confidence_score:.2f})")
```

## New in v0.2.0: Predicate Taxonomies

Normalize predicates to canonical forms using embedding similarity:

```python
from statement_extractor import PredicateTaxonomy, ExtractionOptions

taxonomy = PredicateTaxonomy(predicates=[
    "acquired", "founded", "works_for", "announced",
    "invested_in", "partnered_with"
])

options = ExtractionOptions(predicate_taxonomy=taxonomy)
result = extract_statements(text, options)

# "bought" -> "acquired" via embedding similarity
for stmt in result:
    if stmt.canonical_predicate:
        print(f"{stmt.predicate} -> {stmt.canonical_predicate}")
```

## New in v0.2.2: Contextualized Matching

Predicate canonicalization and deduplication now use **contextualized matching**:
- Compares full "Subject Predicate Object" strings against source text
- Better accuracy because predicates are evaluated in context
- When duplicates are found, keeps the statement with the best match to source text

This means "Apple bought Beats" vs "Apple acquired Beats" are compared holistically, not just "bought" vs "acquired".

## New in v0.2.3: Entity Type Merging & Reversal Detection

### Entity Type Merging

When deduplicating statements, entity types are now automatically merged. If one statement has `UNKNOWN` type and a duplicate has a specific type (like `ORG` or `PERSON`), the specific type is preserved:

```python
# Before deduplication:
# Statement 1: AtlasBio Labs (UNKNOWN) --sued by--> CuraPharm (ORG)
# Statement 2: AtlasBio Labs (ORG) --sued by--> CuraPharm (ORG)

# After deduplication:
# Single statement: AtlasBio Labs (ORG) --sued by--> CuraPharm (ORG)
```

### Subject-Object Reversal Detection

The library now detects when subject and object may have been extracted in the wrong order by comparing embeddings against source text:

```python
from statement_extractor import PredicateComparer

comparer = PredicateComparer()

# Automatically detect and fix reversals
fixed_statements = comparer.detect_and_fix_reversals(statements)

for stmt in fixed_statements:
    if stmt.was_reversed:
        print(f"Fixed reversal: {stmt}")
```

**How it works:**
1. For each statement with source text, compares:
   - "Subject Predicate Object" embedding vs source text
   - "Object Predicate Subject" embedding vs source text
2. If the reversed form has higher similarity, swaps subject and object
3. Sets `was_reversed=True` to indicate the correction

During deduplication, reversed duplicates (e.g., "A -> P -> B" and "B -> P -> A") are now detected and merged, with the correct orientation determined by source text similarity.

## New in v0.4.0: GLiNER2 Integration

v0.4.0 replaces spaCy with **GLiNER2** (205M params) for entity recognition and relation extraction. GLiNER2 is a unified model that handles NER, text classification, structured data extraction, and relation extraction with CPU-optimized inference.

### Why GLiNER2?

The T5-Gemma model excels at:
- **Triple isolation** - identifying that a relationship exists
- **Coreference resolution** - resolving pronouns to named entities

GLiNER2 now handles:
- **Entity recognition** - refining subject/object boundaries
- **Relation extraction** - when predefined predicates are provided
- **Entity scoring** - scoring how "entity-like" subjects/objects are

### Two Extraction Modes

**Mode 1: With Predicate List** (GLiNER2 relation extraction)
```python
from statement_extractor import extract_statements, ExtractionOptions

options = ExtractionOptions(predicates=["works_for", "founded", "acquired", "headquartered_in"])
result = extract_statements("John works for Apple Inc. in Cupertino.", options)
```

Or via CLI:
```bash
corp-extractor "John works for Apple Inc." --predicates "works_for,founded,acquired"
```

**Mode 2: Without Predicate List** (entity-refined extraction)
```python
result = extract_statements("Apple announced a new iPhone.")
# Uses GLiNER2 for entity extraction to refine boundaries
# Extracts predicate from source text using T5-Gemma's hint
```

### Two Candidate Extraction Methods

For each statement, two candidates are generated and the best is selected:

| Method | Description |
|--------|-------------|
| `hybrid` | Model subject/object + GLiNER2/extracted predicate |
| `gliner` | All components refined by GLiNER2 entity recognition |

```python
for stmt in result:
    print(f"{stmt.subject.text} --[{stmt.predicate}]--> {stmt.object.text}")
    print(f"  Method: {stmt.extraction_method}")  # hybrid, gliner, or model
    print(f"  Confidence: {stmt.confidence_score:.2f}")
```

### Combined Quality Scoring

Confidence scores combine **semantic similarity** and **entity recognition**:

| Component | Weight | Description |
|-----------|--------|-------------|
| Semantic similarity | 50% | Cosine similarity between source text and reassembled triple |
| Subject entity score | 25% | How entity-like the subject is (via GLiNER2) |
| Object entity score | 25% | How entity-like the object is (via GLiNER2) |

**Entity scoring (via GLiNER2):**
- Recognized entity with high confidence: 1.0
- Recognized entity with moderate confidence: 0.8
- Partially recognized: 0.6
- Not recognized: 0.2

### Extraction Method Tracking

Each statement includes an `extraction_method` field:
- `hybrid` - Model subject/object + GLiNER2 predicate
- `gliner` - All components refined by GLiNER2 entity recognition
- `model` - All components from T5-Gemma model (only when `--no-gliner`)

### Best Triple Selection

By default, only the **highest-scoring triple** is kept for each source sentence.

To keep all candidate triples:
```python
options = ExtractionOptions(all_triples=True)
result = extract_statements(text, options)
```

Or via CLI:
```bash
corp-extractor "Your text" --all-triples --verbose
```

**Disable GLiNER2 extraction** to use only model output:
```python
options = ExtractionOptions(use_gliner_extraction=False)
result = extract_statements(text, options)
```

Or via CLI:
```bash
corp-extractor "Your text" --no-gliner
```

## Disable Embeddings

```python
options = ExtractionOptions(
    embedding_dedup=False,  # Use exact text matching
    merge_beams=False,      # Select single best beam
)
result = extract_statements(text, options)
```

## Output Formats

```python
from statement_extractor import (
    extract_statements,
    extract_statements_as_json,
    extract_statements_as_xml,
    extract_statements_as_dict,
)

# Pydantic models (default)
result = extract_statements(text)

# JSON string
json_output = extract_statements_as_json(text)

# Raw XML (model's native format)
xml_output = extract_statements_as_xml(text)

# Python dictionary
dict_output = extract_statements_as_dict(text)
```

## Batch Processing

```python
from statement_extractor import StatementExtractor

extractor = StatementExtractor(device="cuda")  # or "mps" (Apple Silicon) or "cpu"

texts = ["Text 1...", "Text 2...", "Text 3..."]
for text in texts:
    result = extractor.extract(text)
    print(f"Found {len(result)} statements")
```

## Entity Types

| Type | Description | Example |
|------|-------------|---------|
| `ORG` | Organizations | Apple Inc., United Nations |
| `PERSON` | People | Tim Cook, Elon Musk |
| `GPE` | Geopolitical entities | USA, California, Paris |
| `LOC` | Non-GPE locations | Mount Everest, Pacific Ocean |
| `PRODUCT` | Products | iPhone, Model S |
| `EVENT` | Events | World Cup, CES 2024 |
| `WORK_OF_ART` | Creative works | Mona Lisa, Game of Thrones |
| `LAW` | Legal documents | GDPR, Clean Air Act |
| `DATE` | Dates | 2024, January 15 |
| `MONEY` | Monetary values | $50 million, €100 |
| `PERCENT` | Percentages | 25%, 0.5% |
| `QUANTITY` | Quantities | 500 employees, 1.5 tons |
| `UNKNOWN` | Unrecognized | (fallback) |

## How It Works

This library uses the T5-Gemma 2 statement extraction model with **Diverse Beam Search** ([Vijayakumar et al., 2016](https://arxiv.org/abs/1610.02424)):

1. **Diverse Beam Search**: Generates 4+ candidate outputs using beam groups with diversity penalty
2. **Quality Scoring**: Each triple scored for groundedness in source text
3. **Beam Merging**: Top beams combined for better coverage
4. **Embedding Dedup**: Semantic similarity removes near-duplicate predicates
5. **Predicate Normalization**: Optional taxonomy matching via embeddings
6. **Contextualized Matching**: Full statement context used for canonicalization and dedup
7. **Entity Type Merging**: UNKNOWN types merged with specific types during dedup
8. **Reversal Detection**: Subject-object reversals detected and corrected via embedding comparison
9. **GLiNER2 Extraction** *(v0.4.0)*: Entity recognition and relation extraction for improved accuracy

## Requirements

- Python 3.10+
- PyTorch 2.0+
- Transformers 5.0+
- Pydantic 2.0+
- sentence-transformers 2.2+
- GLiNER2 (model downloaded automatically on first use)
- ~2GB VRAM (GPU) or ~4GB RAM (CPU)

## Links

- [Model on HuggingFace](https://huggingface.co/Corp-o-Rate-Community/statement-extractor)
- [Web Demo](https://statement-extractor.corp-o-rate.com)
- [Diverse Beam Search Paper](https://arxiv.org/abs/1610.02424)
- [Corp-o-Rate](https://corp-o-rate.com)

## License

MIT License - see LICENSE file for details.

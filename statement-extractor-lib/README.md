# Corp Extractor

Extract structured subject-predicate-object statements from unstructured text using the T5-Gemma 2 model.

[![PyPI version](https://img.shields.io/pypi/v/corp-extractor.svg)](https://pypi.org/project/corp-extractor/)
[![Python 3.10+](https://img.shields.io/pypi/pyversions/corp-extractor.svg)](https://pypi.org/project/corp-extractor/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- **Structured Extraction**: Converts unstructured text into subject-predicate-object triples
- **Entity Type Recognition**: Identifies 12 entity types (ORG, PERSON, GPE, LOC, PRODUCT, EVENT, etc.)
- **Quality Scoring** *(v0.2.0)*: Each triple scored for groundedness (0-1) based on source text
- **Beam Merging** *(v0.2.0)*: Combines top beams for better coverage instead of picking one
- **Embedding-based Dedup** *(v0.2.0)*: Uses semantic similarity to detect near-duplicate predicates
- **Predicate Taxonomies** *(v0.2.0)*: Normalize predicates to canonical forms via embeddings
- **Contextualized Matching** *(v0.2.2)*: Compares full "Subject Predicate Object" against source text for better accuracy
- **Entity Type Merging** *(v0.2.3)*: Automatically merges UNKNOWN entity types with specific types during deduplication
- **Reversal Detection** *(v0.2.3)*: Detects and corrects subject-object reversals using embedding comparison
- **Multiple Output Formats**: Get results as Pydantic models, JSON, XML, or dictionaries

## Installation

```bash
# Recommended: include embedding support for smart deduplication
pip install corp-extractor[embeddings]

# Minimal installation (no embedding features)
pip install corp-extractor
```

**Note**: For GPU support, install PyTorch with CUDA first:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install corp-extractor[embeddings]
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

## Disable Embeddings (Faster, No Extra Dependencies)

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

extractor = StatementExtractor(device="cuda")  # or "cpu"

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
2. **Quality Scoring** *(v0.2.0)*: Each triple scored for groundedness in source text
3. **Beam Merging** *(v0.2.0)*: Top beams combined for better coverage
4. **Embedding Dedup** *(v0.2.0)*: Semantic similarity removes near-duplicate predicates
5. **Predicate Normalization** *(v0.2.0)*: Optional taxonomy matching via embeddings
6. **Contextualized Matching** *(v0.2.2)*: Full statement context used for canonicalization and dedup
7. **Entity Type Merging** *(v0.2.3)*: UNKNOWN types merged with specific types during dedup
8. **Reversal Detection** *(v0.2.3)*: Subject-object reversals detected and corrected via embedding comparison

## Requirements

- Python 3.10+
- PyTorch 2.0+
- Transformers 4.35+
- Pydantic 2.0+
- sentence-transformers 2.2+ *(optional, for embedding features)*
- ~2GB VRAM (GPU) or ~4GB RAM (CPU)

## Links

- [Model on HuggingFace](https://huggingface.co/Corp-o-Rate-Community/statement-extractor)
- [Web Demo](https://statement-extractor.corp-o-rate.com)
- [Diverse Beam Search Paper](https://arxiv.org/abs/1610.02424)
- [Corp-o-Rate](https://corp-o-rate.com)

## License

MIT License - see LICENSE file for details.

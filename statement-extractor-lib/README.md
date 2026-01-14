# Corp Extractor

Extract structured subject-predicate-object statements from unstructured text using the T5-Gemma 2 model.

[![PyPI version](https://badge.fury.io/py/corp-extractor.svg)](https://badge.fury.io/py/corp-extractor)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- **Structured Extraction**: Converts unstructured text into subject-predicate-object triples
- **Entity Type Recognition**: Identifies 12 entity types (ORG, PERSON, GPE, LOC, PRODUCT, EVENT, etc.)
- **High-Quality Output**: Uses [Diverse Beam Search](https://arxiv.org/abs/1610.02424) to generate multiple candidates
- **Smart Retry Logic**: Automatically retries extraction if output quality is below threshold
- **Multiple Output Formats**: Get results as Pydantic models, JSON, XML, or dictionaries

## Installation

```bash
pip install corp-extractor
```

**Note**: Requires PyTorch. For GPU support, install PyTorch with CUDA first:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install corp-extractor
```

## Quick Start

```python
from statement_extractor import extract_statements

# Extract statements from text
result = extract_statements("""
    Apple Inc. announced the iPhone 15 at their September event.
    Tim Cook presented the new features to customers worldwide.
""")

# Iterate over extracted statements
for stmt in result:
    print(f"{stmt.subject.text} ({stmt.subject.type}) "
          f"--[{stmt.predicate}]--> "
          f"{stmt.object.text} ({stmt.object.type})")
```

Output:
```
Apple Inc. (ORG) --[announced]--> iPhone 15 (PRODUCT)
Tim Cook (PERSON) --[presented]--> new features (UNKNOWN)
```

## Output Formats

```python
from statement_extractor import (
    extract_statements,
    extract_statements_as_json,
    extract_statements_as_xml,
    extract_statements_as_dict,
)

text = "Microsoft acquired GitHub in 2018."

# Pydantic models (default)
result = extract_statements(text)
for stmt in result.statements:
    print(stmt.subject, stmt.predicate, stmt.object)

# JSON string
json_output = extract_statements_as_json(text)
print(json_output)

# Raw XML (model's native format)
xml_output = extract_statements_as_xml(text)
print(xml_output)

# Python dictionary
dict_output = extract_statements_as_dict(text)
print(dict_output)
```

## Advanced Usage

### Custom Extraction Options

```python
from statement_extractor import extract_statements, ExtractionOptions

options = ExtractionOptions(
    num_beams=8,              # More beams = more diverse candidates
    diversity_penalty=1.5,    # Higher = more diversity between beams
    max_new_tokens=4096,      # Max tokens to generate
    min_statement_ratio=0.5,  # Min statements per sentence
    max_attempts=5,           # Retry attempts for under-extraction
    deduplicate=True,         # Remove duplicate statements
)

result = extract_statements("Your text here...", options=options)
```

### Using the Extractor Class

For better performance when processing multiple texts:

```python
from statement_extractor import StatementExtractor

# Create extractor once
extractor = StatementExtractor(
    model_id="Corp-o-Rate-Community/statement-extractor",
    device="cuda",  # or "cpu"
)

# Process multiple texts
texts = ["Text 1...", "Text 2...", "Text 3..."]
for text in texts:
    result = extractor.extract(text)
    print(f"Found {len(result)} statements")
```

## Pydantic Models

The library provides fully-typed Pydantic models:

```python
from statement_extractor import Statement, Entity, EntityType, ExtractionResult

# Access statement properties
stmt: Statement = result.statements[0]
print(stmt.subject.text)      # "Apple Inc."
print(stmt.subject.type)      # EntityType.ORG
print(stmt.predicate)         # "announced"
print(stmt.object.text)       # "iPhone 15"
print(stmt.source_text)       # Original sentence (if available)

# Convert to simple tuples
triples = result.to_triples()
# [("Apple Inc.", "announced", "iPhone 15"), ...]
```

### Entity Types

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

This library uses the T5-Gemma 2 statement extraction model with **Diverse Beam Search** ([Vijayakumar et al., 2016](https://arxiv.org/abs/1610.02424)) to generate high-quality extractions:

1. **Diverse Beam Search**: Generates 4+ candidate outputs using beam groups with diversity penalty
2. **Quality-Based Retry**: If extraction count is below threshold, automatically retries
3. **Deduplication**: Removes duplicate statements based on subject-predicate-object triples
4. **Best Selection**: Selects the longest valid output (typically most complete)

## Requirements

- Python 3.10+
- PyTorch 2.0+
- Transformers 4.35+
- Pydantic 2.0+
- ~2GB VRAM (GPU) or ~4GB RAM (CPU)

## Links

- [Model on HuggingFace](https://huggingface.co/Corp-o-Rate-Community/statement-extractor)
- [Web Demo](https://statement-extractor.corp-o-rate.com)
- [Diverse Beam Search Paper](https://arxiv.org/abs/1610.02424)
- [Corp-o-Rate](https://corp-o-rate.com)

## License

MIT License - see LICENSE file for details.
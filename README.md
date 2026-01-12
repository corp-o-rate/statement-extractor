# Statement Extractor

A web demo for the T5-Gemma 2 statement extraction model from [corp-o-rate.com](https://corp-o-rate.com).

## Features

- **Statement Extraction**: Transform unstructured text into structured subject-predicate-object triples
- **Entity Recognition**: Automatic identification of entity types (ORG, PERSON, GPE, EVENT, etc.)
- **Relationship Graph**: Interactive D3.js visualization of entity relationships
- **Coreference Resolution**: Pronouns are resolved to their referenced entities
- **Multiple Usage Options**: API, Python, TypeScript, or run locally

## Quick Start

### Online Demo

Visit [statement-extractor.vercel.app](https://statement-extractor.vercel.app) to try the demo.

### Run Locally

```bash
# Clone the repository
git clone https://github.com/corp-o-rate/statement-extractor
cd statement-extractor

# Install dependencies
pnpm install

# Start the dev server
pnpm dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

## Model Information

- **Architecture**: T5-Gemma 2 (540M parameters)
- **Training Data**: 77,515 examples from corporate and news documents
- **Final Eval Loss**: 0.209
- **Input Format**: Text wrapped in `<page>` tags
- **Output Format**: XML with extracted statements

### HuggingFace Model

The model is available on HuggingFace: [Corp-o-Rate-Community/statement-extractor](https://huggingface.co/Corp-o-Rate-Community/statement-extractor)

## Usage

### Python

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

model = AutoModelForSeq2SeqLM.from_pretrained(
    "Corp-o-Rate-Community/statement-extractor",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(
    "Corp-o-Rate-Community/statement-extractor",
    trust_remote_code=True,
)

text = "Apple Inc. announced a commitment to carbon neutrality by 2030."
inputs = tokenizer(f"<page>{text}</page>", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=2048, num_beams=4)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)
```

### Output Format

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

### Entity Types

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

## Local Server

For unlimited usage without API rate limits, run the model locally using [uv](https://github.com/astral-sh/uv):

```bash
cd local-server
cp .env.example .env  # Edit to set MODEL_PATH
uv sync
uv run python server.py
```

See [local-server/README.md](local-server/README.md) for details.

## Upload Model to HuggingFace

```bash
cd scripts
cp .env.example .env  # Set HF_TOKEN
uv sync
uv run python upload_model.py
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `HF_TOKEN` | HuggingFace API token (for cloud inference) |
| `HF_MODEL` | Model ID (default: `Corp-o-Rate-Community/statement-extractor`) |
| `LOCAL_MODEL_URL` | Local server URL (e.g., `http://localhost:8000`) |

## Tech Stack

- [Next.js](https://nextjs.org/) - React framework
- [Tailwind CSS](https://tailwindcss.com/) - Styling
- [D3.js](https://d3js.org/) - Graph visualization
- [uv](https://github.com/astral-sh/uv) - Python package manager
- [HuggingFace Transformers](https://huggingface.co/docs/transformers) - Model inference
- [Vercel](https://vercel.com/) - Deployment

## About corp-o-rate

Statement Extractor is part of [corp-o-rate.com](https://corp-o-rate.com) - an AI-powered platform for ESG analysis and corporate accountability. Our models extract structured statements from corporate reports, identifying claims, commitments, and impacts.

## License

MIT License

## Contributing

Contributions are welcome! Please open an issue or pull request.
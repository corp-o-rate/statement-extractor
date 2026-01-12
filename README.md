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

## Deployment Options

### RunPod Serverless (Recommended for Production)

Deploy to [RunPod](https://runpod.io) for scalable, pay-per-use GPU inference (~$0.0002/sec).

```bash
cd runpod

# Build and push Docker image (--platform flag required on Mac)
docker build --platform linux/amd64 -t statement-extractor-runpod .
docker tag statement-extractor-runpod YOUR_USERNAME/statement-extractor-runpod
docker push YOUR_USERNAME/statement-extractor-runpod
```

Then on RunPod:
1. Go to [runpod.io/console/serverless](https://www.runpod.io/console/serverless)
2. Click **New Endpoint**
3. Set container image to your pushed image
4. Select GPU (RTX 3090+ recommended)
5. Set Active Workers: 0, Max Workers: 1-3

Call the API:
```bash
curl -X POST https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input": {"text": "<page>Your text here</page>"}}'
```

**Pricing** (pay only when processing):
| Usage | Monthly Cost |
|-------|--------------|
| 100 req/day | ~$0.19 |
| 1,000 req/day | ~$1.86 |
| Idle | $0 |

See [runpod/README.md](runpod/README.md) for detailed instructions.

### Local Server

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
| `RUNPOD_ENDPOINT_ID` | RunPod endpoint ID (recommended for production) |
| `RUNPOD_API_KEY` | RunPod API key |
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
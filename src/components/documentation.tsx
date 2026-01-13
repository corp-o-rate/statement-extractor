'use client';

import { useState } from 'react';
import { Copy, Check, ExternalLink, Terminal, Code2, Server, Cloud } from 'lucide-react';
import { toast } from 'sonner';

type TabId = 'python' | 'typescript' | 'runpod' | 'local' | 'output';

const TABS: { id: TabId; label: string; icon: React.ReactNode }[] = [
  { id: 'python', label: 'Python', icon: <Code2 className="w-4 h-4" /> },
  { id: 'typescript', label: 'TypeScript', icon: <Code2 className="w-4 h-4" /> },
  { id: 'runpod', label: 'RunPod', icon: <Cloud className="w-4 h-4" /> },
  { id: 'local', label: 'Run Locally', icon: <Server className="w-4 h-4" /> },
  { id: 'output', label: 'Output Format', icon: <Terminal className="w-4 h-4" /> },
];

const HF_MODEL = 'Corp-o-Rate-Community/statement-extractor';

const CODE_SNIPPETS: Record<TabId, string> = {
  python: `# Installation
pip install transformers torch

# Usage
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

# Load model and tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained(
    "${HF_MODEL}",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(
    "${HF_MODEL}",
    trust_remote_code=True,
)

# Move to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

def extract_statements(text: str) -> str:
    """Extract statements from text."""
    # Wrap text in page tags
    inputs = tokenizer(
        f"<page>{text}</page>",
        return_tensors="pt",
        max_length=4096,
        truncation=True
    ).to(device)

    # Generate
    outputs = model.generate(
        **inputs,
        max_new_tokens=2048,
        num_beams=4,
        do_sample=False,
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example usage
text = "Apple Inc. announced a commitment to carbon neutrality by 2030."
result = extract_statements(text)
print(result)`,

  typescript: `// Installation
// npm install @huggingface/inference

import { HfInference } from "@huggingface/inference";

const hf = new HfInference(process.env.HF_TOKEN);

async function extractStatements(text: string): Promise<string> {
  // Note: For seq2seq models, you may need to use a custom endpoint
  // or run locally since HF Inference API has limited seq2seq support
  const response = await hf.textGeneration({
    model: "${HF_MODEL}",
    inputs: \`<page>\${text}</page>\`,
    parameters: {
      max_new_tokens: 2048,
      num_beams: 4,
    },
  });

  return response.generated_text;
}

// For production use, we recommend RunPod or running locally
// See the "RunPod" or "Run Locally" tabs for setup instructions

// Example usage
const text = "Apple Inc. announced a commitment to carbon neutrality by 2030.";
const result = await extractStatements(text);
console.log(result);`,

  runpod: `# Deploy to RunPod Serverless (~$0.0002/sec GPU)

## 1. Build and push Docker image
\`\`\`bash
git clone https://github.com/corp-o-rate/statement-extractor
cd statement-extractor/runpod

# Build the image (--platform flag required on Mac)
docker build --platform linux/amd64 -t statement-extractor-runpod .

# Push to Docker Hub
docker tag statement-extractor-runpod YOUR_USERNAME/statement-extractor-runpod
docker push YOUR_USERNAME/statement-extractor-runpod
\`\`\`

## 2. Create RunPod Endpoint
1. Go to runpod.io/console/serverless?ref=sjoylkgj
2. Click "New Endpoint"
3. Set container image to your pushed image
4. Select GPU (RTX 3090+ recommended)
5. Set Active Workers: 0, Max Workers: 1-3

## 3. Call the API
\`\`\`bash
curl -X POST https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync \\
  -H "Authorization: Bearer YOUR_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{"input": {"text": "<page>Your text here</page>"}}'
\`\`\`

## TypeScript Client
\`\`\`typescript
const response = await fetch(
  \`https://api.runpod.ai/v2/\${ENDPOINT_ID}/runsync\`,
  {
    method: 'POST',
    headers: {
      'Authorization': \`Bearer \${RUNPOD_API_KEY}\`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      input: { text: \`<page>\${text}</page>\` },
    }),
  }
);
const data = await response.json();
const output = data.output.output; // XML statements
\`\`\`

## Pricing (pay only when processing)
- RTX 3090: ~$0.00031/sec (~$1.12/hr active)
- Idle: $0 (scales to zero)
- 1000 requests/day: ~$1.86/month`,

  local: `# Run Locally (No API Limits)

## 1. Clone the demo site
\`\`\`bash
git clone https://github.com/corp-o-rate/statement-extractor
cd statement-extractor
pnpm install
\`\`\`

## 2. Install uv and download the model
\`\`\`bash
# Install uv (Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Download the model
uv run huggingface-cli download ${HF_MODEL} --local-dir ./model
\`\`\`

## 3. Run the local API server
\`\`\`bash
cd local-server
cp .env.example .env  # Edit .env to set MODEL_PATH
uv sync
uv run python server.py
\`\`\`

## 4. Start the frontend
\`\`\`bash
# Point to local API
echo "LOCAL_MODEL_URL=http://localhost:8000" >> .env.local
pnpm dev
\`\`\`

## Hardware Requirements
- **Minimum**: 8GB RAM, CPU-only (slow, ~30s per extraction)
- **Recommended**: 16GB RAM + CUDA GPU (fast, ~2s per extraction)
- **Model size**: ~1.5GB disk space`,

  output: `<!-- Output Format -->
<!-- The model outputs XML with extracted statements -->

<statements>
  <stmt>
    <subject type="ORG">Apple Inc.</subject>
    <object type="EVENT">carbon neutrality by 2030</object>
    <predicate>committed to</predicate>
    <text>Apple Inc. committed to achieving carbon neutrality by 2030.</text>
  </stmt>
  <stmt>
    <subject type="PERSON">Tim Cook</subject>
    <object type="MONEY">$4.7 billion</object>
    <predicate>announced investment of</predicate>
    <text>Tim Cook announced an investment of $4.7 billion.</text>
  </stmt>
</statements>

<!-- Entity Types -->
ORG       - Organizations (companies, agencies)
PERSON    - People (names, titles)
GPE       - Geopolitical entities (countries, cities)
LOC       - Locations (mountains, rivers)
PRODUCT   - Products (devices, services)
EVENT     - Events (announcements, meetings)
WORK_OF_ART - Creative works (reports, books)
LAW       - Legal documents
DATE      - Dates and time periods
MONEY     - Monetary values
PERCENT   - Percentages
QUANTITY  - Quantities and measurements`,
};

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    await navigator.clipboard.writeText(text);
    setCopied(true);
    toast.success('Copied to clipboard');
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <button
      onClick={handleCopy}
      className="copy-btn"
      title="Copy to clipboard"
    >
      {copied ? (
        <>
          <Check className="w-4 h-4 text-green-500" />
          Copied!
        </>
      ) : (
        <>
          <Copy className="w-4 h-4" />
          Copy
        </>
      )}
    </button>
  );
}

export function Documentation() {
  const [activeTab, setActiveTab] = useState<TabId>('python');

  return (
    <div id="documentation">
      {/* Tab list */}
      <div className="tab-list overflow-x-auto">
        {TABS.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className="tab-trigger flex items-center gap-2 whitespace-nowrap"
            data-state={activeTab === tab.id ? 'active' : 'inactive'}
          >
            {tab.icon}
            {tab.label}
          </button>
        ))}
      </div>

      {/* Tab content */}
      <div className="mt-4">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <span className="text-sm text-gray-500">Model:</span>
            <a
              href={`https://huggingface.co/${HF_MODEL}`}
              target="_blank"
              rel="noopener noreferrer"
              className="text-sm text-red-600 hover:underline flex items-center gap-1"
            >
              {HF_MODEL}
              <ExternalLink className="w-3 h-3" />
            </a>
          </div>
          <CopyButton text={CODE_SNIPPETS[activeTab]} />
        </div>

        <div className="code-block overflow-x-auto max-h-[500px] overflow-y-auto">
          <pre>
            <code>{CODE_SNIPPETS[activeTab]}</code>
          </pre>
        </div>
      </div>
    </div>
  );
}

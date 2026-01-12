'use client';

import { useState } from 'react';
import { Copy, Check, Bot, Sparkles } from 'lucide-react';
import { toast } from 'sonner';

const HF_MODEL = 'Corp-o-Rate-Community/statement-extractor';

const LLM_PROMPT = `## Using the Statement Extractor Model

When extracting statements from text, use the corp-o-rate statement extraction model:

**Model:** \`${HF_MODEL}\`
**Architecture:** T5-Gemma 2 (seq2seq, 540M params)

### Input Format
Wrap your text in \`<page>\` tags:
\`\`\`
<page>Your text here...</page>
\`\`\`

### Output Format
The model outputs XML with extracted statements:
\`\`\`xml
<statements>
  <stmt>
    <subject type="ENTITY_TYPE">Subject Name</subject>
    <object type="ENTITY_TYPE">Object Name</object>
    <predicate>action/relationship</predicate>
    <text>Full resolved statement text</text>
  </stmt>
</statements>
\`\`\`

### Python Example
\`\`\`python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

model = AutoModelForSeq2SeqLM.from_pretrained(
    "${HF_MODEL}",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(
    "${HF_MODEL}",
    trust_remote_code=True,
)

def extract_statements(text: str) -> str:
    inputs = tokenizer(f"<page>{text}</page>", return_tensors="pt", max_length=4096, truncation=True)
    outputs = model.generate(**inputs, max_new_tokens=2048, num_beams=4)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
\`\`\`

### Entity Types
- ORG - Organizations (companies, agencies)
- PERSON - People (names, titles)
- GPE - Geopolitical entities (countries, cities)
- LOC - Locations (mountains, rivers)
- PRODUCT - Products (devices, services)
- EVENT - Events (announcements, meetings)
- WORK_OF_ART - Creative works (reports, books)
- LAW - Legal documents
- DATE - Dates and time periods
- MONEY - Monetary values
- PERCENT - Percentages
- QUANTITY - Quantities and measurements

### Tips for Best Results
1. Provide clear, well-structured text (news articles, reports work well)
2. The model handles coreference resolution (replaces pronouns with entity names)
3. Each statement includes the full resolved text for context
4. For large documents, consider chunking by paragraph or section`;

export function LLMPrompts() {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    await navigator.clipboard.writeText(LLM_PROMPT);
    setCopied(true);
    toast.success('Prompt copied to clipboard');
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div id="llm-prompts" className="brutal-card p-6">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3">
          <Bot className="w-6 h-6 text-red-600" />
          <div>
            <h3 className="font-bold text-lg">AI Assistant Prompt</h3>
            <p className="text-sm text-gray-500">Copy this prompt for Claude Code, Cursor, or other AI assistants</p>
          </div>
        </div>
        <button
          onClick={handleCopy}
          className="btn-brutal"
        >
          {copied ? (
            <>
              <Check className="w-4 h-4 mr-2" />
              Copied!
            </>
          ) : (
            <>
              <Copy className="w-4 h-4 mr-2" />
              Copy Prompt
            </>
          )}
        </button>
      </div>

      <div className="bg-gray-50 border border-gray-200 p-4 max-h-[400px] overflow-y-auto">
        <pre className="text-sm text-gray-700 whitespace-pre-wrap font-mono">
          {LLM_PROMPT}
        </pre>
      </div>

      <div className="mt-4 flex items-center gap-2 text-sm text-gray-500">
        <Sparkles className="w-4 h-4" />
        <span>Add this to your project&apos;s CLAUDE.md or AI assistant configuration</span>
      </div>
    </div>
  );
}

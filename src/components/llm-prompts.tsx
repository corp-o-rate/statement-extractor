'use client';

import { useState } from 'react';
import { Copy, Check, Bot, Sparkles } from 'lucide-react';
import { toast } from 'sonner';

const LLM_PROMPT = `# SKILL: Statement Extraction with corp-extractor

Use the \`corp-extractor\` Python library to extract structured subject-predicate-object statements from text. Returns Pydantic models with confidence scores.

## Installation

\`\`\`bash
pip install corp-extractor[embeddings]  # Recommended: includes semantic deduplication
\`\`\`

For GPU support, install PyTorch with CUDA first:
\`\`\`bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install corp-extractor[embeddings]
\`\`\`

## Quick Usage

\`\`\`python
from statement_extractor import extract_statements

result = extract_statements("""
    Apple Inc. announced the iPhone 15 at their September event.
    Tim Cook presented the new features to customers worldwide.
""")

for stmt in result:
    print(f"{stmt.subject.text} ({stmt.subject.type})")
    print(f"  --[{stmt.predicate}]--> {stmt.object.text}")
    print(f"  Confidence: {stmt.confidence_score:.2f}")
\`\`\`

## Output Formats

\`\`\`python
from statement_extractor import (
    extract_statements,        # Returns ExtractionResult with Statement objects
    extract_statements_as_json,  # Returns JSON string
    extract_statements_as_xml,   # Returns XML string
    extract_statements_as_dict,  # Returns dict
)
\`\`\`

## Statement Object Structure

Each \`Statement\` has:
- \`subject.text\` - Subject entity text
- \`subject.type\` - Entity type (ORG, PERSON, GPE, etc.)
- \`predicate\` - The relationship/action
- \`object.text\` - Object entity text
- \`object.type\` - Object entity type
- \`source_text\` - Original sentence
- \`confidence_score\` - Groundedness score (0-1)
- \`canonical_predicate\` - Normalized predicate (if taxonomy used)

## Entity Types

ORG, PERSON, GPE (countries/cities), LOC, PRODUCT, EVENT, WORK_OF_ART, LAW, DATE, MONEY, PERCENT, QUANTITY, UNKNOWN

## Precision Mode (Filter Low-Confidence)

\`\`\`python
from statement_extractor import ExtractionOptions, ScoringConfig

options = ExtractionOptions(
    scoring_config=ScoringConfig(min_confidence=0.7)
)
result = extract_statements(text, options)
\`\`\`

## Predicate Taxonomy (Normalize Predicates)

\`\`\`python
from statement_extractor import PredicateTaxonomy, ExtractionOptions

taxonomy = PredicateTaxonomy(predicates=[
    "acquired", "founded", "works_for", "headquartered_in"
])
options = ExtractionOptions(predicate_taxonomy=taxonomy)
result = extract_statements(text, options)

# "bought" -> "acquired" via semantic similarity
for stmt in result:
    if stmt.canonical_predicate:
        print(f"Normalized: {stmt.predicate} -> {stmt.canonical_predicate}")
\`\`\`

## Batch Processing

\`\`\`python
from statement_extractor import StatementExtractor

extractor = StatementExtractor(device="cuda")  # or "cpu"
for text in texts:
    result = extractor.extract(text)
\`\`\`

## Best Practices

1. Use \`[embeddings]\` extra for semantic deduplication
2. Filter by \`confidence_score >= 0.7\` for high precision
3. Use predicate taxonomies for consistent knowledge graphs
4. Process large documents in chunks (by paragraph/section)
5. GPU recommended for production (~2GB VRAM needed)

## Links

- PyPI: https://pypi.org/project/corp-extractor/
- Docs: https://statement-extractor.corp-o-rate.com/docs
- Model: https://huggingface.co/Corp-o-Rate-Community/statement-extractor`;

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
            <h3 className="font-bold text-lg">SKILL.md for AI Assistants</h3>
            <p className="text-sm text-gray-500">Add to your project&apos;s CLAUDE.md or .cursorrules to enable statement extraction</p>
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
        <span>Save as SKILL.md or append to CLAUDE.md in your project root</span>
      </div>
    </div>
  );
}

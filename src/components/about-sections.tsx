'use client';

import { Mail, Heart, Zap, AlertTriangle, RefreshCw, GitBranch, Target, Users, Building2 } from 'lucide-react';

// SVG Diagram: Processing Pipeline
function PipelineDiagram() {
  return (
    <svg viewBox="0 0 800 200" className="w-full max-w-3xl mx-auto my-8">
      {/* Input */}
      <g>
        <rect x="20" y="60" width="120" height="80" rx="8" fill="#f3f4f6" stroke="#d1d5db" strokeWidth="2" />
        <text x="80" y="95" textAnchor="middle" className="fill-gray-700 text-sm font-semibold">Raw Text</text>
        <text x="80" y="115" textAnchor="middle" className="fill-gray-500 text-xs">Unstructured</text>
      </g>

      {/* Arrow 1 */}
      <path d="M145 100 L195 100" stroke="#9ca3af" strokeWidth="2" markerEnd="url(#arrowhead)" />

      {/* Model */}
      <g>
        <rect x="200" y="40" width="140" height="120" rx="8" fill="#fef2f2" stroke="#dc2626" strokeWidth="2" />
        <text x="270" y="75" textAnchor="middle" className="fill-red-700 text-sm font-bold">T5-Gemma 2</text>
        <text x="270" y="95" textAnchor="middle" className="fill-gray-600 text-xs">540M params</text>
        <line x1="220" y1="110" x2="320" y2="110" stroke="#fca5a5" strokeWidth="1" />
        <text x="270" y="130" textAnchor="middle" className="fill-gray-500 text-xs">Beam Search</text>
        <text x="270" y="145" textAnchor="middle" className="fill-gray-500 text-xs">4 candidates</text>
      </g>

      {/* Arrow 2 */}
      <path d="M345 100 L395 100" stroke="#9ca3af" strokeWidth="2" markerEnd="url(#arrowhead)" />

      {/* Handler */}
      <g>
        <rect x="400" y="40" width="140" height="120" rx="8" fill="#f0fdf4" stroke="#16a34a" strokeWidth="2" />
        <text x="470" y="70" textAnchor="middle" className="fill-green-700 text-sm font-bold">Handler</text>
        <text x="470" y="90" textAnchor="middle" className="fill-gray-500 text-xs">Retry logic</text>
        <text x="470" y="105" textAnchor="middle" className="fill-gray-500 text-xs">Deduplication</text>
        <text x="470" y="120" textAnchor="middle" className="fill-gray-500 text-xs">Best selection</text>
        <text x="470" y="135" textAnchor="middle" className="fill-gray-500 text-xs">XML validation</text>
      </g>

      {/* Arrow 3 */}
      <path d="M545 100 L595 100" stroke="#9ca3af" strokeWidth="2" markerEnd="url(#arrowhead)" />

      {/* Output */}
      <g>
        <rect x="600" y="60" width="160" height="80" rx="8" fill="#eff6ff" stroke="#2563eb" strokeWidth="2" />
        <text x="680" y="90" textAnchor="middle" className="fill-blue-700 text-sm font-semibold">Structured Data</text>
        <text x="680" y="110" textAnchor="middle" className="fill-gray-500 text-xs">Subject-Predicate-Object</text>
        <text x="680" y="125" textAnchor="middle" className="fill-gray-500 text-xs">+ Entity Types</text>
      </g>

      {/* Arrowhead marker */}
      <defs>
        <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
          <polygon points="0 0, 10 3.5, 0 7" fill="#9ca3af" />
        </marker>
      </defs>
    </svg>
  );
}

// SVG Diagram: v0.2.0 Processing Flow
function ProcessingFlowDiagram() {
  return (
    <svg viewBox="0 0 700 140" className="w-full max-w-3xl mx-auto my-6">
      {/* Generate */}
      <g>
        <rect x="20" y="40" width="100" height="60" rx="8" fill="#fef2f2" stroke="#dc2626" strokeWidth="2" />
        <text x="70" y="65" textAnchor="middle" className="fill-red-700 text-xs font-bold">Generate</text>
        <text x="70" y="80" textAnchor="middle" className="fill-gray-500 text-xs">4 diverse beams</text>
      </g>

      {/* Arrow 1 */}
      <path d="M125 70 L165 70" stroke="#9ca3af" strokeWidth="2" markerEnd="url(#arrowhead2)" />

      {/* Score */}
      <g>
        <rect x="170" y="40" width="100" height="60" rx="8" fill="#fefce8" stroke="#ca8a04" strokeWidth="2" />
        <text x="220" y="65" textAnchor="middle" className="fill-yellow-700 text-xs font-bold">Score</text>
        <text x="220" y="80" textAnchor="middle" className="fill-gray-500 text-xs">groundedness 0-1</text>
      </g>

      {/* Arrow 2 */}
      <path d="M275 70 L315 70" stroke="#9ca3af" strokeWidth="2" markerEnd="url(#arrowhead2)" />

      {/* Merge Beams */}
      <g>
        <rect x="320" y="40" width="100" height="60" rx="8" fill="#f0fdf4" stroke="#16a34a" strokeWidth="2" />
        <text x="370" y="65" textAnchor="middle" className="fill-green-700 text-xs font-bold">Merge</text>
        <text x="370" y="80" textAnchor="middle" className="fill-gray-500 text-xs">top-N beams</text>
      </g>

      {/* Arrow 3 */}
      <path d="M425 70 L465 70" stroke="#9ca3af" strokeWidth="2" markerEnd="url(#arrowhead2)" />

      {/* Deduplicate */}
      <g>
        <rect x="470" y="40" width="100" height="60" rx="8" fill="#f5f3ff" stroke="#7c3aed" strokeWidth="2" />
        <text x="520" y="65" textAnchor="middle" className="fill-purple-700 text-xs font-bold">Deduplicate</text>
        <text x="520" y="80" textAnchor="middle" className="fill-gray-500 text-xs">embeddings</text>
      </g>

      {/* Arrow 4 */}
      <path d="M575 70 L615 70" stroke="#9ca3af" strokeWidth="2" markerEnd="url(#arrowhead2)" />

      {/* Output */}
      <g>
        <circle cx="650" cy="70" r="35" fill="#eff6ff" stroke="#2563eb" strokeWidth="2" />
        <text x="650" y="75" textAnchor="middle" className="fill-blue-700 text-xs font-bold">Output</text>
      </g>

      <defs>
        <marker id="arrowhead2" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
          <polygon points="0 0, 10 3.5, 0 7" fill="#9ca3af" />
        </marker>
      </defs>
    </svg>
  );
}

// Corp-o-Rate Logo
function CorpORateLogo() {
  return (
    <svg viewBox="0 0 200 60" className="w-40 mx-auto mb-4">
      <text x="100" y="40" textAnchor="middle" className="fill-black text-3xl font-black" style={{ fontFamily: 'system-ui' }}>
        CORP-O-RATE
      </text>
      <line x1="20" y1="50" x2="180" y2="50" stroke="#dc2626" strokeWidth="3" />
    </svg>
  );
}

export function HowItWorks() {
  return (
    <section className="py-16 px-4 sm:px-6 lg:px-8 border-t">
      <div className="max-w-4xl mx-auto">
        <div className="text-center mb-10">
          <span className="section-label">TECHNICAL DETAILS</span>
          <h2 className="text-2xl md:text-3xl font-black mt-4">How It Works</h2>
        </div>

        {/* Pipeline Overview */}
        <div className="mb-12">
          <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
            <GitBranch className="w-5 h-5 text-red-600" />
            Processing Pipeline
          </h3>
          <PipelineDiagram />
          <p className="text-gray-600 text-center max-w-2xl mx-auto">
            Text flows through the T5-Gemma 2 model using{' '}
            <a
              href="https://arxiv.org/abs/1610.02424"
              target="_blank"
              rel="noopener noreferrer"
              className="text-red-600 hover:underline"
            >
              Diverse Beam Search
            </a>{' '}
            (Vijayakumar et al., 2016) to generate multiple candidate extractions,
            then the handler selects and validates the best result.
          </p>
        </div>

        {/* v0.2.0 Processing Flow */}
        <div className="mb-12">
          <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
            <RefreshCw className="w-5 h-5 text-red-600" />
            Quality-First Extraction <span className="text-xs bg-red-100 text-red-700 px-1.5 py-0.5 rounded ml-2">v0.2.0</span>
          </h3>
          <ProcessingFlowDiagram />
          <p className="text-gray-600 text-center max-w-2xl mx-auto mb-6">
            Multiple beam outputs are scored for quality, merged for coverage, and deduplicated using
            semantic similarity (paraphrase-MiniLM-L6-v2, 0.65 threshold).
          </p>
          <div className="grid md:grid-cols-2 gap-6 mt-6">
            <div className="bg-gray-50 p-4 rounded-lg border">
              <h4 className="font-semibold mb-2">
                <a
                  href="https://arxiv.org/abs/1610.02424"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="hover:text-red-600 transition-colors"
                >
                  Diverse Beam Search
                </a>
              </h4>
              <p className="text-sm text-gray-600">
                Uses the <a href="https://arxiv.org/abs/1610.02424" target="_blank" rel="noopener noreferrer" className="text-red-600 hover:underline">Diverse Beam Search algorithm</a> (Vijayakumar et al., 2016)
                to generate 4 different candidate outputs using beam groups with diversity penalty,
                exploring multiple possible interpretations of the text.
              </p>
            </div>
            <div className="bg-gray-50 p-4 rounded-lg border">
              <h4 className="font-semibold mb-2">Quality Scoring <span className="text-xs bg-red-100 text-red-700 px-1 rounded">v0.2</span></h4>
              <p className="text-sm text-gray-600">
                Each triple is scored for groundedness (0-1) based on whether subject/object appear in source text,
                predicate has a lexical trigger, and entities are in proximity.
              </p>
            </div>
            <div className="bg-gray-50 p-4 rounded-lg border">
              <h4 className="font-semibold mb-2">Beam Merging <span className="text-xs bg-red-100 text-red-700 px-1 rounded">v0.2</span></h4>
              <p className="text-sm text-gray-600">
                Instead of picking one beam, merges top-N beams to maximize coverage.
                Resolves duplicates by keeping highest-confidence triples.
              </p>
            </div>
            <div className="bg-gray-50 p-4 rounded-lg border">
              <h4 className="font-semibold mb-2">Embedding Dedup <span className="text-xs bg-red-100 text-red-700 px-1 rounded">v0.2</span></h4>
              <p className="text-sm text-gray-600">
                Uses sentence embeddings to detect semantically similar predicates (&quot;bought&quot; ≈ &quot;acquired&quot;),
                removing redundant triples more intelligently than exact text matching.
              </p>
            </div>
          </div>
        </div>

        {/* Limitations */}
        <div className="mb-12">
          <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
            <AlertTriangle className="w-5 h-5 text-amber-500" />
            Known Limitations
          </h3>
          <div className="bg-amber-50 border border-amber-200 rounded-lg p-6">
            <ul className="space-y-3 text-gray-700">
              <li className="flex gap-3">
                <span className="text-amber-500 font-bold">1.</span>
                <span><strong>Complex sentences:</strong> Very long sentences with multiple nested clauses may result in incomplete extraction or incorrect predicate assignment.</span>
              </li>
              <li className="flex gap-3">
                <span className="text-amber-500 font-bold">2.</span>
                <span><strong>Implicit relationships:</strong> The model works best with explicit statements. Implied or contextual relationships may be missed.</span>
              </li>
              <li className="flex gap-3">
                <span className="text-amber-500 font-bold">3.</span>
                <span><strong>Domain specificity:</strong> Trained primarily on corporate/news text. Performance may vary on highly technical or specialized content.</span>
              </li>
              <li className="flex gap-3">
                <span className="text-amber-500 font-bold">4.</span>
                <span><strong>Coreference limits:</strong> While the model resolves many pronouns, complex anaphora chains or ambiguous references may not resolve correctly.</span>
              </li>
              <li className="flex gap-3">
                <span className="text-amber-500 font-bold">5.</span>
                <span><strong>Entity type coverage:</strong> Some specialized entity types (e.g., scientific terms, technical products) may default to UNKNOWN.</span>
              </li>
            </ul>
          </div>
        </div>

        {/* Areas for Improvement */}
        <div className="mb-12">
          <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
            <Target className="w-5 h-5 text-red-600" />
            Roadmap &amp; Areas for Improvement
          </h3>

          {/* Completed in v0.2.0 */}
          <div className="mb-6 p-4 bg-green-50 border border-green-200 rounded-lg">
            <h4 className="font-semibold text-green-800 mb-2 flex items-center gap-2">
              <span className="text-green-600">✓</span> Completed in v0.2.0
            </h4>
            <ul className="text-sm text-green-700 space-y-1">
              <li><strong>Confidence Scores</strong> — Each triple scored 0-1 for groundedness</li>
              <li><strong>Beam Merging</strong> — Combines top beams for better coverage</li>
              <li><strong>Embedding Dedup</strong> — Semantic similarity for predicate matching</li>
            </ul>
          </div>

          {/* Planned */}
          <div className="grid md:grid-cols-2 gap-4">
            <div className="border-l-4 border-red-600 pl-4 py-2">
              <h4 className="font-semibold">Larger Training Dataset</h4>
              <p className="text-sm text-gray-600">Expanding beyond 77K examples with more diverse sources</p>
            </div>
            <div className="border-l-4 border-red-600 pl-4 py-2">
              <h4 className="font-semibold">Multi-hop Reasoning</h4>
              <p className="text-sm text-gray-600">Better handling of statements that span multiple sentences</p>
            </div>
            <div className="border-l-4 border-red-600 pl-4 py-2">
              <h4 className="font-semibold">Negation Handling</h4>
              <p className="text-sm text-gray-600">Better detection of negative statements and contradictions</p>
            </div>
            <div className="border-l-4 border-red-600 pl-4 py-2">
              <h4 className="font-semibold">Relationship Type Classification</h4>
              <p className="text-sm text-gray-600">Categorize extracted relationships (acquisition, partnership, etc.)</p>
            </div>
          </div>
        </div>

        {/* Feedback CTA */}
        <div className="bg-gradient-to-r from-red-50 to-orange-50 border-2 border-red-200 rounded-lg p-8 text-center">
          <h3 className="text-xl font-bold mb-3 flex items-center justify-center gap-2">
            <Heart className="w-5 h-5 text-red-500" />
            We Need Your Feedback
          </h3>
          <p className="text-gray-700 mb-4 max-w-xl mx-auto">
            This model is actively being improved. If you encounter incorrect extractions, missing statements,
            or have suggestions for improvement, we&apos;d love to hear from you. Use the &quot;Correct&quot; button above
            to submit fixes, or reach out directly.
          </p>
          <a
            href="mailto:neil@corp-o-rate.com"
            className="inline-flex items-center gap-2 px-6 py-3 bg-black text-white font-bold hover:bg-gray-800 transition-colors"
          >
            <Mail className="w-5 h-5" />
            neil@corp-o-rate.com
          </a>
        </div>
      </div>
    </section>
  );
}

export function AboutCorpORate() {
  return (
    <section className="py-16 px-4 sm:px-6 lg:px-8 bg-gray-900 text-white">
      <div className="max-w-4xl mx-auto">
        <div className="text-center mb-10">
          <span className="text-red-400 text-xs font-bold tracking-widest uppercase">Who We Are</span>
          <h2 className="text-2xl md:text-3xl font-black mt-4">About Corp-o-Rate</h2>
          <CorpORateLogo />
        </div>

        {/* Mission */}
        <div className="mb-10">
          <div className="flex items-start gap-4 mb-6">
            <div className="bg-red-600 p-3 rounded-lg shrink-0">
              <Building2 className="w-6 h-6 text-white" />
            </div>
            <div>
              <h3 className="text-xl font-bold mb-2">The Glassdoor of ESG</h3>
              <p className="text-gray-300 text-lg">
                Real corporate intelligence from real people. Track what companies <em>actually do</em>, not what they claim.
              </p>
            </div>
          </div>

          <p className="text-gray-400 mb-4">
            Corp-o-Rate is building a community-powered corporate accountability platform. We believe that glossy
            sustainability reports and PR-polished ESG claims don&apos;t tell the full story. Our mission is to surface
            the truth about corporate behavior through crowdsourced intelligence, AI-powered analysis, and
            transparent data.
          </p>

          <p className="text-gray-400">
            This statement extraction model is one piece of that puzzle &mdash; automatically extracting relationships and meaningful statements
            from research, news, and corporate documents. Available as the{' '}
            <a
              href="https://pypi.org/project/corp-extractor"
              target="_blank"
              rel="noopener noreferrer"
              className="text-red-400 hover:underline"
            >
              corp-extractor
            </a>{' '}
            Python library on PyPI. This is the first part of our analysis and we&apos;ll be releasing other re-usable components
            as we progress.
          </p>
        </div>

        {/* What we're building */}
        <div className="grid md:grid-cols-3 gap-6 mb-10">
          <div className="bg-gray-800 p-5 rounded-lg border border-gray-700">
            <Users className="w-8 h-8 text-red-400 mb-3" />
            <h4 className="font-bold mb-2">Community-Driven</h4>
            <p className="text-sm text-gray-400">
              Powered by employees, consumers, and researchers sharing real knowledge about corporate practices.
            </p>
          </div>
          <div className="bg-gray-800 p-5 rounded-lg border border-gray-700">
            <Zap className="w-8 h-8 text-red-400 mb-3" />
            <h4 className="font-bold mb-2">AI-Powered</h4>
            <p className="text-sm text-gray-400">
              Using NLP and knowledge graphs to structure, connect, and analyze corporate claims at scale.
            </p>
          </div>
          <div className="bg-gray-800 p-5 rounded-lg border border-gray-700">
            <Target className="w-8 h-8 text-red-400 mb-3" />
            <h4 className="font-bold mb-2">100% Independent</h4>
            <p className="text-sm text-gray-400">
              No corporate sponsors. No conflicts of interest. Just transparent corporate intelligence.
            </p>
          </div>
        </div>

        {/* Pre-funding notice */}
        <div className="bg-gradient-to-r from-red-900/50 to-orange-900/50 border border-red-700 rounded-lg p-8 text-center">
          <h3 className="text-xl font-bold mb-3">We&apos;re Pre-Funding &amp; Running on Fumes</h3>
          <p className="text-gray-300 mb-6 max-w-2xl mx-auto">
            Corp-o-Rate is currently bootstrapped and self-funded. We&apos;re building in public, shipping what we can,
            and working toward our mission one step at a time. If you believe in corporate accountability and
            transparent business intelligence, we&apos;d love your support.
          </p>

          <div className="flex flex-wrap justify-center gap-4 mb-6">
            <div className="bg-gray-800 px-4 py-2 rounded border border-gray-600">
              <span className="text-gray-400 text-sm">GPU Credits</span>
              <p className="font-bold text-white">Help us train better models</p>
            </div>
            <div className="bg-gray-800 px-4 py-2 rounded border border-gray-600">
              <span className="text-gray-400 text-sm">Angel Investment</span>
              <p className="font-bold text-white">Help us scale the platform</p>
            </div>
            <div className="bg-gray-800 px-4 py-2 rounded border border-gray-600">
              <span className="text-gray-400 text-sm">Partnerships</span>
              <p className="font-bold text-white">Data, research, or distribution</p>
            </div>
          </div>

          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <a
              href="mailto:neil@corp-o-rate.com?subject=Corp-o-Rate%20Support"
              className="inline-flex items-center justify-center gap-2 px-6 py-3 bg-red-600 text-white font-bold hover:bg-red-700 transition-colors"
            >
              <Mail className="w-5 h-5" />
              Get in Touch
            </a>
            <a
              href="https://corp-o-rate.com"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center justify-center gap-2 px-6 py-3 bg-white text-gray-900 font-bold hover:bg-gray-100 transition-colors"
            >
              Visit Corp-o-Rate
            </a>
          </div>
        </div>

        <p className="text-center text-gray-500 mt-8 text-sm">
          Shop smarter. Invest better. Know which companies match your values.
        </p>
      </div>
    </section>
  );
}

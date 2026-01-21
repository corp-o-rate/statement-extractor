'use client';

import { Mail, Heart, Zap, AlertTriangle, Target, Users, Building2, Layers } from 'lucide-react';
import { PipelineFlowDiagram, DataFlowDiagram } from './pipeline-diagram';

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

        {/* 5-Stage Pipeline Overview */}
        <div className="mb-12">
          <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
            <Layers className="w-5 h-5 text-red-600" />
            5-Stage Pipeline Architecture <span className="text-xs bg-red-100 text-red-700 px-1.5 py-0.5 rounded ml-2">v0.8.0</span>
          </h3>
          <PipelineFlowDiagram />
          <p className="text-gray-600 text-center max-w-2xl mx-auto">
            Text flows through a modular plugin-based pipeline. Each stage transforms the data progressively,
            from raw text to fully qualified, labeled statements with taxonomy classifications.
          </p>
        </div>

        {/* Pipeline Stages Detail */}
        <div className="mb-12">
          <h3 className="text-xl font-bold mb-6">Pipeline Stages</h3>
          <div className="overflow-x-auto">
            <table className="w-full text-sm border-collapse">
              <thead>
                <tr className="bg-gray-100">
                  <th className="border border-gray-300 px-4 py-2 text-left">Stage</th>
                  <th className="border border-gray-300 px-4 py-2 text-left">Name</th>
                  <th className="border border-gray-300 px-4 py-2 text-left">Purpose</th>
                  <th className="border border-gray-300 px-4 py-2 text-left">Key Technology</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td className="border border-gray-300 px-4 py-2 font-bold text-red-600">1</td>
                  <td className="border border-gray-300 px-4 py-2">Splitting</td>
                  <td className="border border-gray-300 px-4 py-2">Text → Atomic Statements</td>
                  <td className="border border-gray-300 px-4 py-2">T5-Gemma2 (540M params) with{' '}
                    <a href="https://arxiv.org/abs/1610.02424" target="_blank" rel="noopener noreferrer" className="text-red-600 hover:underline">Diverse Beam Search</a>
                  </td>
                </tr>
                <tr className="bg-gray-50">
                  <td className="border border-gray-300 px-4 py-2 font-bold text-yellow-600">2</td>
                  <td className="border border-gray-300 px-4 py-2">Extraction</td>
                  <td className="border border-gray-300 px-4 py-2">Atomic Statements → Typed Triples</td>
                  <td className="border border-gray-300 px-4 py-2">GLiNER2 (205M params) entity recognition</td>
                </tr>
                <tr>
                  <td className="border border-gray-300 px-4 py-2 font-bold text-green-600">3</td>
                  <td className="border border-gray-300 px-4 py-2">Qualification</td>
                  <td className="border border-gray-300 px-4 py-2">Entities → Canonical names, identifiers, FQN</td>
                  <td className="border border-gray-300 px-4 py-2">Company embedding database (SEC, GLEIF, UK Companies House)</td>
                </tr>
                <tr className="bg-gray-50">
                  <td className="border border-gray-300 px-4 py-2 font-bold text-purple-600">4</td>
                  <td className="border border-gray-300 px-4 py-2">Labeling</td>
                  <td className="border border-gray-300 px-4 py-2">Add simple classifications</td>
                  <td className="border border-gray-300 px-4 py-2">Multi-choice classifiers (sentiment, relation type)</td>
                </tr>
                <tr>
                  <td className="border border-gray-300 px-4 py-2 font-bold text-orange-600">5</td>
                  <td className="border border-gray-300 px-4 py-2">Taxonomy</td>
                  <td className="border border-gray-300 px-4 py-2">Classify against ESG taxonomy</td>
                  <td className="border border-gray-300 px-4 py-2">MNLI zero-shot <em>or</em> embedding similarity</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        {/* Data Flow */}
        <div className="mb-12">
          <h3 className="text-xl font-bold mb-4">Data Flow</h3>
          <DataFlowDiagram />
          <p className="text-gray-600 text-center max-w-2xl mx-auto">
            Data is progressively enriched through each stage, from raw text to fully qualified statements
            with entity types, canonical names, sentiment labels, and taxonomy classifications.
          </p>
        </div>

        {/* Technical Features */}
        <div className="mb-12">
          <h3 className="text-xl font-bold mb-6">Technical Features</h3>
          <div className="grid md:grid-cols-2 gap-6">
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
                The T5-Gemma2 model uses{' '}
                <a href="https://arxiv.org/abs/1610.02424" target="_blank" rel="noopener noreferrer" className="text-red-600 hover:underline">Diverse Beam Search</a>{' '}
                (Vijayakumar et al., 2016) to generate 4 diverse candidate outputs,
                exploring multiple interpretations of the text.
              </p>
            </div>
            <div className="bg-gray-50 p-4 rounded-lg border">
              <h4 className="font-semibold mb-2">GLiNER2 Entity Extraction</h4>
              <p className="text-sm text-gray-600">
                GLiNER2 (205M params) refines entity boundaries and scores how &quot;entity-like&quot;
                subjects and objects are. Uses 324 default predicates across 21 categories for relation extraction.
              </p>
            </div>
            <div className="bg-gray-50 p-4 rounded-lg border">
              <h4 className="font-semibold mb-2">Entity Qualification</h4>
              <p className="text-sm text-gray-600">
                Company embedding database (~100K+ SEC, ~3M GLEIF, ~5M UK companies) provides fast vector similarity
                search to resolve entities to canonical names with identifiers (LEI, CIK, company numbers) and FQN.
              </p>
            </div>
            <div className="bg-gray-50 p-4 rounded-lg border">
              <h4 className="font-semibold mb-2">Taxonomy Classification</h4>
              <p className="text-sm text-gray-600">
                Statements are classified against an ESG taxonomy using either MNLI zero-shot classification
                or embedding similarity, returning multiple labels above confidence thresholds.
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

          {/* Completed */}
          <div className="mb-6 p-4 bg-green-50 border border-green-200 rounded-lg">
            <h4 className="font-semibold text-green-800 mb-2 flex items-center gap-2">
              <span className="text-green-600">✓</span> Recently Completed
            </h4>
            <ul className="text-sm text-green-700 space-y-1">
              <li><strong>5-Stage Pipeline Architecture</strong> <span className="text-xs">(v0.8.0)</span> — Merged qualification + canonicalization into single stage</li>
              <li><strong>Company Embedding Database</strong> <span className="text-xs">(v0.8.0)</span> — Fast vector search for ~100K+ SEC, ~3M GLEIF, ~5M UK companies</li>
              <li><strong>Taxonomy Classification</strong> <span className="text-xs">(v0.5.0)</span> — MNLI + embedding-based ESG taxonomy classification</li>
              <li><strong>Entity Qualification</strong> <span className="text-xs">(v0.5.0)</span> — LEI, ticker, CIK lookups with canonical names and FQN</li>
              <li><strong>Statement Labeling</strong> <span className="text-xs">(v0.5.0)</span> — Sentiment analysis and relation type classification</li>
              <li><strong>GLiNER2 Integration</strong> <span className="text-xs">(v0.4.0)</span> — 205M param model for entity recognition and relation extraction</li>
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
              <h4 className="font-semibold">Knowledge Graph Integration</h4>
              <p className="text-sm text-gray-600">Link extracted entities to external knowledge bases (Wikidata, etc.)</p>
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

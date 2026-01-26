'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { ArrowLeft, Book, Code2, Settings, Boxes, Rocket, Server, ChevronRight, Terminal, Database } from 'lucide-react';
import { TableOfContents, TocItem } from '@/components/docs/table-of-contents';

// Import documentation sections
import GettingStarted from './sections/getting-started.mdx';
import Cli from './sections/cli.mdx';
import CoreConcepts from './sections/core-concepts.mdx';
import EntityDatabase from './sections/entity-database.mdx';
import ApiReference from './sections/api-reference.mdx';
import Configuration from './sections/configuration.mdx';
import EntityTypes from './sections/entity-types.mdx';
import Examples from './sections/examples.mdx';
import Deployment from './sections/deployment.mdx';

// Table of contents structure
const tocItems: TocItem[] = [
  {
    id: 'getting-started',
    label: 'Getting Started',
    level: 2,
    children: [
      { id: 'installation', label: 'Installation', level: 3 },
      { id: 'quick-start', label: 'Quick Start', level: 3 },
      { id: 'requirements', label: 'Requirements', level: 3 },
    ],
  },
  {
    id: 'cli',
    label: 'Command Line Interface',
    level: 2,
    children: [
      { id: 'commands-overview', label: 'Commands Overview', level: 3 },
      { id: 'split-command', label: 'Split Command', level: 3 },
      { id: 'pipeline-command', label: 'Pipeline Command', level: 3 },
      { id: 'document-commands', label: 'Document Command', level: 3 },
      { id: 'db-commands', label: 'Database Commands', level: 3 },
      { id: 'plugins-command', label: 'Plugins Command', level: 3 },
      { id: 'cli-output', label: 'Output Formats', level: 3 },
    ],
  },
  {
    id: 'core-concepts',
    label: 'Core Concepts',
    level: 2,
    children: [
      { id: 'statement-extraction', label: 'Statement Extraction', level: 3 },
      { id: 'diverse-beam-search', label: 'Diverse Beam Search', level: 3 },
      { id: 'quality-scoring', label: 'Quality Scoring', level: 3 },
      { id: 'gliner2-integration', label: 'GLiNER2 Integration', level: 3 },
      { id: 'pipeline-architecture', label: 'Pipeline Architecture', level: 3 },
      { id: 'document-processing', label: 'Document Processing', level: 3 },
      { id: 'company-database', label: 'Company Database', level: 3 },
    ],
  },
  {
    id: 'entity-database',
    label: 'Entity Database',
    level: 2,
    children: [
      { id: 'entity-db-quickstart', label: 'Quick Start', level: 3 },
      { id: 'getting-database', label: 'Getting the Database', level: 3 },
      { id: 'database-schema', label: 'Database Schema', level: 3 },
      { id: 'entity-db-types', label: 'Entity Types', level: 3 },
      { id: 'data-sources', label: 'Data Sources', level: 3 },
      { id: 'entity-db-python', label: 'Python API', level: 3 },
      { id: 'building-database', label: 'Building Your Own', level: 3 },
      { id: 'canonicalization', label: 'Canonicalization', level: 3 },
    ],
  },
  {
    id: 'api-reference',
    label: 'API Reference',
    level: 2,
    children: [
      { id: 'functions', label: 'Functions', level: 3 },
      { id: 'classes', label: 'Classes', level: 3 },
      { id: 'data-models', label: 'Data Models', level: 3 },
    ],
  },
  {
    id: 'configuration',
    label: 'Configuration',
    level: 2,
    children: [
      { id: 'extraction-options', label: 'ExtractionOptions', level: 3 },
      { id: 'scoring-config', label: 'ScoringConfig', level: 3 },
      { id: 'predicate-config', label: 'PredicateComparisonConfig', level: 3 },
    ],
  },
  {
    id: 'entity-types',
    label: 'Entity Types',
    level: 2,
  },
  {
    id: 'examples',
    label: 'Examples',
    level: 2,
    children: [
      { id: 'basic-extraction', label: 'Basic Extraction', level: 3 },
      { id: 'batch-processing', label: 'Batch Processing', level: 3 },
      { id: 'confidence-filtering', label: 'Confidence Filtering', level: 3 },
      { id: 'predicate-taxonomy', label: 'Predicate Taxonomy', level: 3 },
    ],
  },
  {
    id: 'deployment',
    label: 'Deployment',
    level: 2,
    children: [
      { id: 'local-inference', label: 'Local Inference', level: 3 },
      { id: 'runpod-serverless', label: 'RunPod Serverless', level: 3 },
    ],
  },
];

export default function DocsPage() {
  return (
    <>
      {/* Header */}
      <header className="sticky top-0 z-50 bg-white border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center gap-4">
              <Link
                href="/"
                className="flex items-center gap-2 text-gray-600 hover:text-gray-900 transition-colors"
              >
                <ArrowLeft className="w-4 h-4" />
                <span className="text-sm font-medium">Back to Demo</span>
              </Link>
              <div className="h-6 w-px bg-gray-200" />
              <div className="flex items-center gap-2">
                <Book className="w-5 h-5 text-red-600" />
                <span className="font-bold">Documentation</span>
              </div>
            </div>
            <div className="flex items-center gap-4">
              <a
                href="https://pypi.org/project/corp-extractor"
                target="_blank"
                rel="noopener noreferrer"
                className="text-sm text-gray-600 hover:text-red-600 transition-colors"
              >
                PyPI
              </a>
              <a
                href="https://huggingface.co/Corp-o-Rate-Community/statement-extractor"
                target="_blank"
                rel="noopener noreferrer"
                className="text-sm text-gray-600 hover:text-red-600 transition-colors"
              >
                Model
              </a>
              <a
                href="https://github.com/corp-o-rate/statement-extractor"
                target="_blank"
                rel="noopener noreferrer"
                className="text-sm text-gray-600 hover:text-red-600 transition-colors"
              >
                GitHub
              </a>
            </div>
          </div>
        </div>
      </header>

      {/* Main content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="lg:grid lg:grid-cols-[1fr_250px] lg:gap-8">
          {/* Documentation content */}
          <main className="prose prose-gray max-w-none">
            {/* Hero */}
            <div className="not-prose mb-12">
              <span className="text-red-600 text-xs font-bold tracking-widest uppercase">
                corp-extractor v0.7.0
              </span>
              <h1 className="text-4xl font-black mt-2 mb-4">
                Statement Extractor Documentation
              </h1>
              <p className="text-xl text-gray-600 max-w-2xl">
                Extract structured subject-predicate-object statements from unstructured text
                using T5-Gemma 2 and GLiNER2 models with document processing, entity resolution, and taxonomy classification.
              </p>

              {/* Quick links */}
              <div className="grid sm:grid-cols-2 lg:grid-cols-5 gap-4 mt-8">
                <a
                  href="#getting-started"
                  className="flex items-center gap-3 p-4 border-2 border-gray-200 hover:border-red-500 transition-colors group"
                >
                  <Rocket className="w-5 h-5 text-gray-400 group-hover:text-red-500" />
                  <div>
                    <div className="font-semibold">Getting Started</div>
                    <div className="text-sm text-gray-500">Installation & quick start</div>
                  </div>
                </a>
                <a
                  href="#cli"
                  className="flex items-center gap-3 p-4 border-2 border-gray-200 hover:border-red-500 transition-colors group"
                >
                  <Terminal className="w-5 h-5 text-gray-400 group-hover:text-red-500" />
                  <div>
                    <div className="font-semibold">CLI</div>
                    <div className="text-sm text-gray-500">Command line & documents</div>
                  </div>
                </a>
                <a
                  href="#pipeline-architecture"
                  className="flex items-center gap-3 p-4 border-2 border-gray-200 hover:border-red-500 transition-colors group"
                >
                  <Server className="w-5 h-5 text-gray-400 group-hover:text-red-500" />
                  <div>
                    <div className="font-semibold">5-Stage Pipeline</div>
                    <div className="text-sm text-gray-500">Entity resolution & taxonomy</div>
                  </div>
                </a>
                <a
                  href="#entity-database"
                  className="flex items-center gap-3 p-4 border-2 border-gray-200 hover:border-red-500 transition-colors group"
                >
                  <Database className="w-5 h-5 text-gray-400 group-hover:text-red-500" />
                  <div>
                    <div className="font-semibold">Entity Database</div>
                    <div className="text-sm text-gray-500">Organizations & people</div>
                  </div>
                </a>
                <a
                  href="#api-reference"
                  className="flex items-center gap-3 p-4 border-2 border-gray-200 hover:border-red-500 transition-colors group"
                >
                  <Code2 className="w-5 h-5 text-gray-400 group-hover:text-red-500" />
                  <div>
                    <div className="font-semibold">API Reference</div>
                    <div className="text-sm text-gray-500">Functions & classes</div>
                  </div>
                </a>
              </div>
            </div>

            {/* Documentation sections */}
            <GettingStarted />
            <Cli />
            <CoreConcepts />
            <EntityDatabase />
            <ApiReference />
            <Configuration />
            <EntityTypes />
            <Examples />
            <Deployment />
          </main>

          {/* Table of contents sidebar */}
          <aside className="hidden lg:block">
            <TableOfContents items={tocItems} />
          </aside>
        </div>
      </div>

      {/* Footer */}
      <footer className="border-t border-gray-200 mt-16 py-8">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between text-sm text-gray-500">
            <div>
              Built by{' '}
              <a href="https://corp-o-rate.com" className="text-red-600 hover:underline">
                Corp-o-Rate
              </a>
            </div>
            <div>MIT License</div>
          </div>
        </div>
      </footer>
    </>
  );
}

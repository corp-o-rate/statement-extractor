'use client';

import { useState, useEffect } from 'react';
import { ChevronDown, ChevronUp, Layers, Loader2 } from 'lucide-react';

// Type for topic config from the JSON
interface TopicConfig {
  description: string;
  id: number;
  mnli_label: string;
  embedding_label: string;
}

// Type for the API response
type TaxonomyResponse = Record<string, Record<string, TopicConfig>>;

// Transform category ID to display name
function formatCategoryName(id: string): string {
  return id
    .split('_')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
}

// Get category color class
function getCategoryColor(id: string): string {
  const colors: Record<string, string> = {
    environment: 'bg-green-50 border-green-200',
    society: 'bg-blue-50 border-blue-200',
    governance: 'bg-purple-50 border-purple-200',
    animals: 'bg-amber-50 border-amber-200',
    industry: 'bg-slate-50 border-slate-200',
    human_harm: 'bg-red-50 border-red-200',
    human_benefit: 'bg-emerald-50 border-emerald-200',
    animal_harm: 'bg-orange-50 border-orange-200',
    animal_benefit: 'bg-teal-50 border-teal-200',
    environment_harm: 'bg-rose-50 border-rose-200',
    environment_benefit: 'bg-lime-50 border-lime-200',
  };
  return colors[id] || 'bg-gray-50 border-gray-200';
}

// Get topic badge color
function getTopicColor(categoryId: string): string {
  const colors: Record<string, string> = {
    environment: 'bg-green-100 border-green-300 text-green-700',
    society: 'bg-blue-100 border-blue-300 text-blue-700',
    governance: 'bg-purple-100 border-purple-300 text-purple-700',
    animals: 'bg-amber-100 border-amber-300 text-amber-700',
    industry: 'bg-slate-100 border-slate-300 text-slate-700',
    human_harm: 'bg-red-100 border-red-300 text-red-700',
    human_benefit: 'bg-emerald-100 border-emerald-300 text-emerald-700',
    animal_harm: 'bg-orange-100 border-orange-300 text-orange-700',
    animal_benefit: 'bg-teal-100 border-teal-300 text-teal-700',
    environment_harm: 'bg-rose-100 border-rose-300 text-rose-700',
    environment_benefit: 'bg-lime-100 border-lime-300 text-lime-700',
  };
  return colors[categoryId] || 'bg-gray-100 border-gray-200 text-gray-600';
}

export function StatementTaxonomy() {
  const [isExpanded, setIsExpanded] = useState(false);
  const [expandedCategory, setExpandedCategory] = useState<string | null>(null);
  const [taxonomy, setTaxonomy] = useState<TaxonomyResponse | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch('/api/taxonomy')
      .then(res => {
        if (!res.ok) throw new Error('Failed to load taxonomy');
        return res.json();
      })
      .then(data => {
        setTaxonomy(data);
        setIsLoading(false);
      })
      .catch(err => {
        console.error('Failed to load taxonomy:', err);
        setError(err.message);
        setIsLoading(false);
      });
  }, []);

  // Calculate totals
  const categories = taxonomy ? Object.entries(taxonomy) : [];
  const totalTopics = categories.reduce(
    (sum, [, topics]) => sum + Object.keys(topics).length,
    0
  );

  // Get preview topics (first few from first few categories)
  const previewTopics: { name: string; description: string; category: string }[] = [];
  if (taxonomy) {
    for (const [categoryId, topics] of categories.slice(0, 4)) {
      for (const [name, config] of Object.entries(topics).slice(0, 3)) {
        if (previewTopics.length < 12) {
          previewTopics.push({ name, description: config.description, category: categoryId });
        }
      }
    }
  }

  const toggleCategory = (categoryId: string) => {
    setExpandedCategory(expandedCategory === categoryId ? null : categoryId);
  };

  return (
    <div id="statement-taxonomy" className="scroll-mt-24">
      <div className="editorial-card p-6">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <Layers className="w-5 h-5 text-red-600" />
            <h3 className="font-bold text-lg">Statement Taxonomy</h3>
            {!isLoading && !error && (
              <span className="text-sm text-gray-500">
                ({totalTopics} topics in {categories.length} categories)
              </span>
            )}
          </div>
          <button
            onClick={() => setIsExpanded(!isExpanded)}
            className="inline-flex items-center gap-1 text-sm text-gray-600 hover:text-black transition-colors"
            disabled={isLoading || !!error}
          >
            {isExpanded ? (
              <>
                <ChevronUp className="w-4 h-4" />
                Collapse
              </>
            ) : (
              <>
                <ChevronDown className="w-4 h-4" />
                Expand
              </>
            )}
          </button>
        </div>

        <p className="text-gray-600 mb-4">
          Stage 6 of the pipeline classifies statements against this ESG taxonomy using
          embedding similarity or MNLI inference. Each topic includes descriptions to guide
          classification. You can provide a custom taxonomy via{' '}
          <code className="text-xs bg-gray-100 px-1 py-0.5 rounded">taxonomy_file</code>{' '}
          parameter to the taxonomy classifier plugins.
        </p>

        {isLoading ? (
          <div className="flex items-center justify-center py-8 text-gray-500">
            <Loader2 className="w-5 h-5 animate-spin mr-2" />
            Loading taxonomy...
          </div>
        ) : error ? (
          <div className="text-red-600 py-4">
            Failed to load taxonomy: {error}
          </div>
        ) : isExpanded ? (
          <div className="space-y-3">
            {categories.map(([categoryId, topics]) => (
              <div
                key={categoryId}
                className={`rounded border ${getCategoryColor(categoryId)}`}
              >
                <button
                  onClick={() => toggleCategory(categoryId)}
                  className="w-full p-3 flex items-center justify-between text-left hover:bg-white/50 transition-colors"
                >
                  <div className="flex items-center gap-2">
                    <h4 className="font-semibold text-sm text-gray-700">
                      {formatCategoryName(categoryId)}
                    </h4>
                    <span className="text-xs text-gray-500">
                      ({Object.keys(topics).length} topics)
                    </span>
                  </div>
                  {expandedCategory === categoryId ? (
                    <ChevronUp className="w-4 h-4 text-gray-400" />
                  ) : (
                    <ChevronDown className="w-4 h-4 text-gray-400" />
                  )}
                </button>
                {expandedCategory === categoryId && (
                  <div className="px-3 pb-3">
                    <div className="flex flex-wrap gap-1">
                      {Object.entries(topics).map(([name, config]) => (
                        <span
                          key={name}
                          className={`text-xs px-2 py-0.5 border rounded cursor-help ${getTopicColor(categoryId)}`}
                          title={config.description}
                        >
                          {name}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        ) : (
          <div className="flex flex-wrap gap-1">
            {previewTopics.map((topic) => (
              <span
                key={`${topic.category}-${topic.name}`}
                className={`text-xs px-2 py-0.5 border rounded cursor-help ${getTopicColor(topic.category)}`}
                title={topic.description}
              >
                {topic.name}
              </span>
            ))}
            <span className="text-xs px-2 py-0.5 text-gray-400">
              ...and {totalTopics - previewTopics.length} more
            </span>
          </div>
        )}

        {/* Override instructions */}
        {isExpanded && (
          <div className="mt-4 p-3 bg-gray-50 rounded border text-sm text-gray-600">
            <strong>Custom taxonomy:</strong> Create a JSON file with the same structure
            and pass it to the taxonomy classifier:
            <pre className="mt-2 bg-gray-800 text-gray-100 p-2 rounded text-xs overflow-x-auto">
{`from statement_extractor.plugins.taxonomy.embedding import EmbeddingTaxonomyClassifier

classifier = EmbeddingTaxonomyClassifier(taxonomy_file="my_taxonomy.json")`}
            </pre>
          </div>
        )}
      </div>
    </div>
  );
}
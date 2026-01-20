'use client';

import { useState, useEffect } from 'react';
import { ChevronDown, ChevronUp, Tags, Loader2 } from 'lucide-react';

// Type for predicate config from the JSON
interface PredicateConfig {
  description: string;
  threshold: number;
}

// Type for the API response
type PredicatesResponse = Record<string, Record<string, PredicateConfig>>;

// Transform category ID to display name
function formatCategoryName(id: string): string {
  return id
    .split('_')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
}

// Transform predicate name for display
function formatPredicateName(name: string): string {
  return name.replace(/_/g, ' ');
}

export function CanonicalPredicates() {
  const [isExpanded, setIsExpanded] = useState(false);
  const [predicates, setPredicates] = useState<PredicatesResponse | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch('/api/predicates')
      .then(res => {
        if (!res.ok) throw new Error('Failed to load predicates');
        return res.json();
      })
      .then(data => {
        setPredicates(data);
        setIsLoading(false);
      })
      .catch(err => {
        console.error('Failed to load predicates:', err);
        setError(err.message);
        setIsLoading(false);
      });
  }, []);

  // Calculate totals
  const categories = predicates ? Object.entries(predicates) : [];
  const totalPredicates = categories.reduce(
    (sum, [, preds]) => sum + Object.keys(preds).length,
    0
  );

  // Get preview predicates (first few from first few categories)
  const previewPredicates: { name: string; description: string }[] = [];
  if (predicates) {
    for (const [, preds] of categories.slice(0, 3)) {
      for (const [name, config] of Object.entries(preds).slice(0, 4)) {
        if (previewPredicates.length < 12) {
          previewPredicates.push({ name, description: config.description });
        }
      }
    }
  }

  return (
    <div id="canonical-predicates" className="scroll-mt-24">
      <div className="editorial-card p-6">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <Tags className="w-5 h-5 text-red-600" />
            <h3 className="font-bold text-lg">Default Predicates</h3>
            {!isLoading && !error && (
              <span className="text-sm text-gray-500">
                ({totalPredicates} predicates in {categories.length} categories)
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
          The extractor uses GLiNER2 relation extraction with these default predicates.
          Each predicate has a confidence threshold (typically 0.65-0.8) that filters
          low-confidence matches. You can override these defaults by providing a custom{' '}
          <code className="text-xs bg-gray-100 px-1 py-0.5 rounded">predicates_file</code>{' '}
          parameter to the <code className="text-xs bg-gray-100 px-1 py-0.5 rounded">GLiNER2Extractor</code>.
        </p>

        {isLoading ? (
          <div className="flex items-center justify-center py-8 text-gray-500">
            <Loader2 className="w-5 h-5 animate-spin mr-2" />
            Loading predicates...
          </div>
        ) : error ? (
          <div className="text-red-600 py-4">
            Failed to load predicates: {error}
          </div>
        ) : isExpanded ? (
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
            {categories.map(([categoryId, preds]) => (
              <div key={categoryId} className="bg-gray-50 p-3 rounded border">
                <h4 className="font-semibold text-sm text-gray-700 mb-2">
                  {formatCategoryName(categoryId)}
                  <span className="font-normal text-gray-400 ml-1">
                    ({Object.keys(preds).length})
                  </span>
                </h4>
                <div className="flex flex-wrap gap-1">
                  {Object.entries(preds).map(([name, config]) => (
                    <span
                      key={name}
                      className="text-xs px-2 py-0.5 bg-white border border-gray-200 rounded text-gray-600 cursor-help"
                      title={config.description}
                    >
                      {formatPredicateName(name)}
                    </span>
                  ))}
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="flex flex-wrap gap-1">
            {previewPredicates.map((pred) => (
              <span
                key={pred.name}
                className="text-xs px-2 py-0.5 bg-gray-100 border border-gray-200 rounded text-gray-600 cursor-help"
                title={pred.description}
              >
                {formatPredicateName(pred.name)}
              </span>
            ))}
            <span className="text-xs px-2 py-0.5 text-gray-400">
              ...and {totalPredicates - previewPredicates.length} more
            </span>
          </div>
        )}

        {/* Override instructions */}
        {isExpanded && (
          <div className="mt-4 p-3 bg-gray-50 rounded border text-sm text-gray-600">
            <strong>Custom predicates:</strong> Create a JSON file with the same structure
            and pass it to the extractor:
            <pre className="mt-2 bg-gray-800 text-gray-100 p-2 rounded text-xs overflow-x-auto">
{`from statement_extractor.plugins.extractors.gliner2 import GLiNER2Extractor

extractor = GLiNER2Extractor(predicates_file="my_predicates.json")`}
            </pre>
          </div>
        )}
      </div>
    </div>
  );
}
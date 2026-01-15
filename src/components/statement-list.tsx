'use client';

import { Statement, ExtractionMethod, getEntityBadgeClass } from '@/lib/types';
import { ArrowRight, Quote, ThumbsUp, Loader2 } from 'lucide-react';

interface StatementListProps {
  statements: Statement[];
  onLike?: () => void;
  isLiking?: boolean;
  hasLiked?: boolean;
}

function EntityBadge({ name, type }: { name: string; type: string }) {
  const badgeClass = getEntityBadgeClass(type as any);

  return (
    <span className={`badge ${badgeClass}`}>
      <span className="font-normal mr-1 opacity-70">{type}</span>
      {name}
    </span>
  );
}

function ConfidenceBadge({ confidence }: { confidence?: number }) {
  // Show dash if no confidence
  if (confidence === undefined || confidence === null) {
    return (
      <span
        className="text-xs font-medium px-1.5 py-0.5 rounded bg-gray-100 text-gray-500"
        title="Confidence not available"
      >
        —%
      </span>
    );
  }

  // Color based on confidence level
  const percent = Math.round(confidence * 100);
  let colorClass = 'bg-gray-100 text-gray-600';
  if (confidence >= 0.8) {
    colorClass = 'bg-green-100 text-green-700';
  } else if (confidence >= 0.6) {
    colorClass = 'bg-yellow-100 text-yellow-700';
  } else {
    colorClass = 'bg-red-100 text-red-700';
  }

  return (
    <span
      className={`text-xs font-medium px-1.5 py-0.5 rounded ${colorClass}`}
      title={`Confidence: ${percent}%`}
    >
      {percent}%
    </span>
  );
}

function ExtractionMethodBadge({ method }: { method?: ExtractionMethod }) {
  const methodLabels: Record<ExtractionMethod, string> = {
    hybrid: 'Hybrid',
    spacy: 'spaCy',
    split: 'Split',
    model: 'Model',
  };

  const methodColors: Record<ExtractionMethod, string> = {
    hybrid: 'bg-blue-100 text-blue-700',
    spacy: 'bg-purple-100 text-purple-700',
    split: 'bg-orange-100 text-orange-700',
    model: 'bg-gray-100 text-gray-600',
  };

  const methodDescriptions: Record<ExtractionMethod, string> = {
    hybrid: 'Model subject/object + spaCy predicate',
    spacy: 'All components from spaCy parsing',
    split: 'Source text split around predicate',
    model: 'All components from T5-Gemma model',
  };

  // Show "Unknown" if no method
  if (!method) {
    return (
      <span
        className="text-xs font-medium px-1.5 py-0.5 rounded bg-gray-100 text-gray-500"
        title="Extraction method not specified"
      >
        —
      </span>
    );
  }

  return (
    <span
      className={`text-xs font-medium px-1.5 py-0.5 rounded ${methodColors[method]}`}
      title={methodDescriptions[method]}
    >
      {methodLabels[method]}
    </span>
  );
}

function StatementCard({ statement, index }: { statement: Statement; index: number }) {
  return (
    <div className="editorial-card p-4">
      {/* Header row with statement number and confidence */}
      <div className="flex items-start justify-between gap-3 mb-3">
        <div className="flex items-start gap-3 flex-1 min-w-0">
          <span className="text-xs font-bold text-gray-400 mt-1">#{index + 1}</span>
          <div className="flex-1 min-w-0">
            {/* Subject → Predicate → Object */}
            <div className="flex flex-wrap items-center gap-2">
              <EntityBadge name={statement.subject.name} type={statement.subject.type} />
              <ArrowRight className="w-4 h-4 text-gray-400 flex-shrink-0" />
              <span className="font-semibold text-gray-700">
                {statement.canonicalPredicate || statement.predicate}
                {statement.canonicalPredicate && statement.canonicalPredicate !== statement.predicate && (
                  <span className="text-gray-400 font-normal ml-1" title={`Original: "${statement.predicate}"`}>
                    *
                  </span>
                )}
              </span>
              {statement.object.name && (
                <>
                  <ArrowRight className="w-4 h-4 text-gray-400 flex-shrink-0" />
                  <EntityBadge name={statement.object.name} type={statement.object.type} />
                </>
              )}
            </div>
          </div>
        </div>
        {/* Method and confidence badges in top right */}
        <div className="flex items-center gap-2 text-xs">
          <div className="flex items-center gap-1">
            <span className="text-gray-400">Source:</span>
            <ExtractionMethodBadge method={statement.extractionMethod} />
          </div>
          <div className="flex items-center gap-1">
            <span className="text-gray-400">Conf:</span>
            <ConfidenceBadge confidence={statement.confidence} />
          </div>
        </div>
      </div>

      {/* Full statement text */}
      {statement.text && (
        <div className="flex items-start gap-2 mt-3 pl-6">
          <Quote className="w-4 h-4 text-gray-300 flex-shrink-0 mt-0.5" />
          <p className="text-sm text-gray-600 italic leading-relaxed">
            {statement.text}
          </p>
        </div>
      )}
    </div>
  );
}

export function StatementList({ statements, onLike, isLiking, hasLiked }: StatementListProps) {
  if (statements.length === 0) {
    return (
      <div className="text-center py-12 text-gray-500">
        <p>No statements extracted yet.</p>
        <p className="text-sm mt-2">Enter some text and click &quot;Extract Statements&quot; to begin.</p>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between mb-4">
        <h3 className="font-bold text-lg">
          Extracted Statements
        </h3>
        <span className="text-sm text-gray-500">
          {statements.length} statement{statements.length !== 1 ? 's' : ''}
        </span>
      </div>

      {statements.map((statement, index) => (
        <StatementCard key={index} statement={statement} index={index} />
      ))}

      {/* Like button */}
      {onLike && (
        <div className="pt-4 border-t mt-4">
          <button
            onClick={onLike}
            disabled={isLiking || hasLiked}
            className={`inline-flex items-center gap-2 px-4 py-2 text-sm font-semibold transition-all ${
              hasLiked
                ? 'text-green-600 bg-green-50 border border-green-200 cursor-default'
                : isLiking
                ? 'text-gray-400 bg-gray-50 border border-gray-200 cursor-wait'
                : 'text-gray-600 hover:text-green-600 hover:bg-green-50 border border-gray-200 hover:border-green-200'
            }`}
            title={hasLiked ? 'Thanks for the feedback!' : 'Mark extraction as correct'}
          >
            {isLiking ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <ThumbsUp className={`w-4 h-4 ${hasLiked ? 'fill-current' : ''}`} />
            )}
            {hasLiked ? 'Saved!' : 'Looks good'}
          </button>
        </div>
      )}
    </div>
  );
}

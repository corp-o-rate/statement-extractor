'use client';

import { Statement, getEntityBadgeClass } from '@/lib/types';
import { ArrowRight, Quote } from 'lucide-react';

interface StatementListProps {
  statements: Statement[];
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

function StatementCard({ statement, index }: { statement: Statement; index: number }) {
  return (
    <div className="editorial-card p-4">
      {/* Statement number */}
      <div className="flex items-start gap-3 mb-3">
        <span className="text-xs font-bold text-gray-400 mt-1">#{index + 1}</span>
        <div className="flex-1">
          {/* Subject → Predicate → Object */}
          <div className="flex flex-wrap items-center gap-2">
            <EntityBadge name={statement.subject.name} type={statement.subject.type} />
            <ArrowRight className="w-4 h-4 text-gray-400 flex-shrink-0" />
            <span className="font-semibold text-gray-700">{statement.predicate}</span>
            {statement.object.name && (
              <>
                <ArrowRight className="w-4 h-4 text-gray-400 flex-shrink-0" />
                <EntityBadge name={statement.object.name} type={statement.object.type} />
              </>
            )}
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

export function StatementList({ statements }: StatementListProps) {
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
    </div>
  );
}

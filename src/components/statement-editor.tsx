'use client';

import { Statement, EntityType, getEntityBadgeClass } from '@/lib/types';
import { Trash2, Plus } from 'lucide-react';

const ENTITY_TYPES: EntityType[] = [
  'ORG', 'PERSON', 'GPE', 'LOC', 'PRODUCT', 'EVENT',
  'WORK_OF_ART', 'LAW', 'DATE', 'MONEY', 'PERCENT', 'QUANTITY', 'UNKNOWN'
];

interface StatementEditorProps {
  statements: Statement[];
  onChange: (statements: Statement[]) => void;
  onSubmit: () => void;
  isSubmitting: boolean;
  hasChanges: boolean;
}

function EditableStatement({
  statement,
  index,
  onUpdate,
  onDelete,
}: {
  statement: Statement;
  index: number;
  onUpdate: (updated: Statement) => void;
  onDelete: () => void;
}) {
  return (
    <div className="editorial-card p-4 space-y-3">
      <div className="flex items-start justify-between gap-2">
        <span className="text-xs font-bold text-gray-400">#{index + 1}</span>
        <button
          onClick={onDelete}
          className="p-1 text-gray-400 hover:text-red-600 transition-colors"
          title="Delete statement"
        >
          <Trash2 className="w-4 h-4" />
        </button>
      </div>

      {/* Subject */}
      <div className="grid grid-cols-[80px_1fr] gap-2 items-center">
        <label className="text-xs font-semibold text-gray-500">Subject</label>
        <div className="flex gap-2">
          <select
            value={statement.subject.type}
            onChange={(e) => onUpdate({
              ...statement,
              subject: { ...statement.subject, type: e.target.value as EntityType }
            })}
            className="px-2 py-1 text-xs border rounded bg-white"
          >
            {ENTITY_TYPES.map(t => <option key={t} value={t}>{t}</option>)}
          </select>
          <input
            type="text"
            value={statement.subject.name}
            onChange={(e) => onUpdate({
              ...statement,
              subject: { ...statement.subject, name: e.target.value }
            })}
            className="flex-1 px-2 py-1 text-sm border rounded"
            placeholder="Subject name"
          />
        </div>
      </div>

      {/* Predicate */}
      <div className="grid grid-cols-[80px_1fr] gap-2 items-center">
        <label className="text-xs font-semibold text-gray-500">Predicate</label>
        <input
          type="text"
          value={statement.predicate}
          onChange={(e) => onUpdate({ ...statement, predicate: e.target.value })}
          className="px-2 py-1 text-sm border rounded"
          placeholder="Predicate (verb/relationship)"
        />
      </div>

      {/* Object */}
      <div className="grid grid-cols-[80px_1fr] gap-2 items-center">
        <label className="text-xs font-semibold text-gray-500">Object</label>
        <div className="flex gap-2">
          <select
            value={statement.object.type}
            onChange={(e) => onUpdate({
              ...statement,
              object: { ...statement.object, type: e.target.value as EntityType }
            })}
            className="px-2 py-1 text-xs border rounded bg-white"
          >
            {ENTITY_TYPES.map(t => <option key={t} value={t}>{t}</option>)}
          </select>
          <input
            type="text"
            value={statement.object.name}
            onChange={(e) => onUpdate({
              ...statement,
              object: { ...statement.object, name: e.target.value }
            })}
            className="flex-1 px-2 py-1 text-sm border rounded"
            placeholder="Object name (optional)"
          />
        </div>
      </div>

      {/* Full text */}
      <div className="grid grid-cols-[80px_1fr] gap-2 items-start">
        <label className="text-xs font-semibold text-gray-500 pt-1">Text</label>
        <textarea
          value={statement.text}
          onChange={(e) => onUpdate({ ...statement, text: e.target.value })}
          className="px-2 py-1 text-sm border rounded resize-none"
          rows={2}
          placeholder="Full resolved statement text"
        />
      </div>
    </div>
  );
}

export function StatementEditor({
  statements,
  onChange,
  onSubmit,
  isSubmitting,
  hasChanges,
}: StatementEditorProps) {
  const handleUpdate = (index: number, updated: Statement) => {
    const newStatements = [...statements];
    newStatements[index] = updated;
    onChange(newStatements);
  };

  const handleDelete = (index: number) => {
    onChange(statements.filter((_, i) => i !== index));
  };

  const handleAdd = () => {
    onChange([
      ...statements,
      {
        subject: { name: '', type: 'UNKNOWN' },
        object: { name: '', type: 'UNKNOWN' },
        predicate: '',
        text: '',
      },
    ]);
  };

  if (statements.length === 0) {
    return (
      <div className="text-center py-8">
        <p className="text-gray-500 mb-4">No statements to edit.</p>
        <button
          onClick={handleAdd}
          className="inline-flex items-center gap-2 px-4 py-2 text-sm font-semibold border-2 border-dashed border-gray-300 rounded hover:border-gray-400 transition-colors"
        >
          <Plus className="w-4 h-4" />
          Add Statement
        </button>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="font-bold text-lg">Edit Statements</h3>
        <span className="text-sm text-gray-500">
          {statements.length} statement{statements.length !== 1 ? 's' : ''}
        </span>
      </div>

      <div className="space-y-3 max-h-[500px] overflow-y-auto pr-2">
        {statements.map((stmt, index) => (
          <EditableStatement
            key={index}
            statement={stmt}
            index={index}
            onUpdate={(updated) => handleUpdate(index, updated)}
            onDelete={() => handleDelete(index)}
          />
        ))}
      </div>

      <div className="flex items-center justify-between pt-4 border-t">
        <button
          onClick={handleAdd}
          className="inline-flex items-center gap-2 px-3 py-2 text-sm font-semibold text-gray-600 hover:text-black transition-colors"
        >
          <Plus className="w-4 h-4" />
          Add Statement
        </button>

        <button
          onClick={onSubmit}
          disabled={isSubmitting || !hasChanges}
          className={`px-6 py-2 font-bold text-white transition-all ${
            hasChanges && !isSubmitting
              ? 'bg-red-600 hover:bg-red-700'
              : 'bg-gray-300 cursor-not-allowed'
          }`}
        >
          {isSubmitting ? 'Submitting...' : 'Submit Correction'}
        </button>
      </div>
    </div>
  );
}

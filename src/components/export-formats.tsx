'use client';

import { useState } from 'react';
import { Statement } from '@/lib/types';
import { Copy, Check, FileDown } from 'lucide-react';

type ExportFormat = 'csv' | 'json' | 'xml' | 'cypher';

interface ExportFormatsProps {
  statements: Statement[];
}

function statementsToCSV(statements: Statement[]): string {
  const headers = ['subject', 'subject_type', 'predicate', 'canonical_predicate', 'object', 'object_type', 'text', 'confidence'];
  const rows = statements.map(s => [
    `"${s.subject.name.replace(/"/g, '""')}"`,
    s.subject.type,
    `"${s.predicate.replace(/"/g, '""')}"`,
    s.canonicalPredicate ? `"${s.canonicalPredicate.replace(/"/g, '""')}"` : '',
    `"${s.object.name.replace(/"/g, '""')}"`,
    s.object.type,
    `"${s.text.replace(/"/g, '""')}"`,
    s.confidence !== undefined ? s.confidence.toFixed(3) : '',
  ].join(','));
  return [headers.join(','), ...rows].join('\n');
}

function statementsToJSON(statements: Statement[]): string {
  return JSON.stringify(statements, null, 2);
}

function statementsToXML(statements: Statement[]): string {
  const stmtElements = statements.map(s => {
    const confidenceAttr = s.confidence !== undefined ? ` confidence="${s.confidence.toFixed(3)}"` : '';
    const canonicalEl = s.canonicalPredicate ? `\n    <canonical_predicate>${escapeXML(s.canonicalPredicate)}</canonical_predicate>` : '';
    return `  <stmt${confidenceAttr}>
    <subject type="${s.subject.type}">${escapeXML(s.subject.name)}</subject>
    <predicate>${escapeXML(s.predicate)}</predicate>${canonicalEl}
    <object type="${s.object.type}">${escapeXML(s.object.name)}</object>
    <text>${escapeXML(s.text)}</text>
  </stmt>`;
  }).join('\n');
  return `<statements>\n${stmtElements}\n</statements>`;
}

function escapeXML(str: string): string {
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&apos;');
}

function statementsToCypher(statements: Statement[]): string {
  const nodes = new Map<string, { name: string; type: string }>();

  // Collect unique nodes
  statements.forEach(s => {
    const subjectKey = `${s.subject.name}:${s.subject.type}`;
    const objectKey = `${s.object.name}:${s.object.type}`;
    if (!nodes.has(subjectKey)) {
      nodes.set(subjectKey, { name: s.subject.name, type: s.subject.type });
    }
    if (!nodes.has(objectKey)) {
      nodes.set(objectKey, { name: s.object.name, type: s.object.type });
    }
  });

  // Generate node creation statements
  const nodeStatements = Array.from(nodes.entries()).map(([, node], idx) => {
    const varName = `n${idx}`;
    const escapedName = node.name.replace(/'/g, "\\'");
    return `CREATE (${varName}:${node.type} {name: '${escapedName}'})`;
  });

  // Create a map from node key to variable name
  const nodeVarMap = new Map<string, string>();
  Array.from(nodes.keys()).forEach((nodeKey, idx) => {
    nodeVarMap.set(nodeKey, `n${idx}`);
  });

  // Generate relationship statements
  const relStatements = statements.map(s => {
    const subjectKey = `${s.subject.name}:${s.subject.type}`;
    const objectKey = `${s.object.name}:${s.object.type}`;
    const subjectVar = nodeVarMap.get(subjectKey);
    const objectVar = nodeVarMap.get(objectKey);
    const relType = (s.canonicalPredicate || s.predicate).toUpperCase().replace(/[^A-Z0-9]/g, '_');
    const escapedPredicate = s.predicate.replace(/'/g, "\\'");
    const confidenceProp = s.confidence !== undefined ? `, confidence: ${s.confidence.toFixed(3)}` : '';
    const canonicalProp = s.canonicalPredicate ? `, canonical_predicate: '${s.canonicalPredicate.replace(/'/g, "\\'")}'` : '';
    return `CREATE (${subjectVar})-[:${relType} {predicate: '${escapedPredicate}'${canonicalProp}${confidenceProp}}]->(${objectVar})`;
  });

  return [...nodeStatements, '', ...relStatements].join('\n');
}

export function ExportFormats({ statements }: ExportFormatsProps) {
  const [format, setFormat] = useState<ExportFormat>('json');
  const [copied, setCopied] = useState(false);

  const formats: { id: ExportFormat; label: string }[] = [
    { id: 'json', label: 'JSON' },
    { id: 'csv', label: 'CSV' },
    { id: 'xml', label: 'XML' },
    { id: 'cypher', label: 'Cypher' },
  ];

  const getFormattedOutput = (): string => {
    if (statements.length === 0) {
      return '// No statements to export';
    }
    switch (format) {
      case 'csv':
        return statementsToCSV(statements);
      case 'json':
        return statementsToJSON(statements);
      case 'xml':
        return statementsToXML(statements);
      case 'cypher':
        return statementsToCypher(statements);
    }
  };

  const output = getFormattedOutput();

  const handleCopy = async () => {
    await navigator.clipboard.writeText(output);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleDownload = () => {
    const extensions: Record<ExportFormat, string> = {
      csv: 'csv',
      json: 'json',
      xml: 'xml',
      cypher: 'cypher',
    };
    const mimeTypes: Record<ExportFormat, string> = {
      csv: 'text/csv',
      json: 'application/json',
      xml: 'application/xml',
      cypher: 'text/plain',
    };
    const blob = new Blob([output], { type: mimeTypes[format] });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `statements.${extensions[format]}`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="editorial-card p-4 md:p-6">
      <div className="flex flex-wrap items-center justify-between gap-4 mb-4">
        <div className="flex gap-2">
          {formats.map(f => (
            <button
              key={f.id}
              onClick={() => setFormat(f.id)}
              className={`px-3 py-1.5 text-sm font-semibold border-2 transition-colors ${
                format === f.id
                  ? 'bg-black text-white border-black'
                  : 'bg-white text-gray-600 border-gray-300 hover:border-black'
              }`}
            >
              {f.label}
            </button>
          ))}
        </div>
        <div className="flex gap-2">
          <button
            onClick={handleCopy}
            disabled={statements.length === 0}
            className="inline-flex items-center gap-2 px-3 py-1.5 text-sm font-semibold text-gray-600 hover:text-black border rounded transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {copied ? (
              <>
                <Check className="w-4 h-4" />
                Copied
              </>
            ) : (
              <>
                <Copy className="w-4 h-4" />
                Copy
              </>
            )}
          </button>
          <button
            onClick={handleDownload}
            disabled={statements.length === 0}
            className="inline-flex items-center gap-2 px-3 py-1.5 text-sm font-semibold text-gray-600 hover:text-black border rounded transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <FileDown className="w-4 h-4" />
            Download
          </button>
        </div>
      </div>
      <pre className="bg-gray-50 border-2 border-gray-200 p-4 overflow-x-auto text-sm font-mono max-h-64 overflow-y-auto">
        <code>{output}</code>
      </pre>
    </div>
  );
}

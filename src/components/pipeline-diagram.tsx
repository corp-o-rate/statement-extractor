'use client';

// Beautiful custom pipeline diagram - no library dependencies, just clean React + CSS

const stages = [
  { num: 1, name: 'Splitting', tech: 'T5-Gemma2', color: 'red' },
  { num: 2, name: 'Extraction', tech: 'GLiNER2', color: 'amber' },
  { num: 3, name: 'Qualification', tech: 'Gemma 1B + APIs', color: 'emerald' },
  { num: 4, name: 'Canonicalization', tech: 'Fuzzy Match', color: 'blue' },
  { num: 5, name: 'Labeling', tech: 'Multi-choice', color: 'violet' },
  { num: 6, name: 'Taxonomy', tech: 'MNLI / Embed', color: 'orange' },
];

const colorMap: Record<string, { bg: string; border: string; text: string; dot: string }> = {
  red: { bg: 'bg-red-50', border: 'border-red-500', text: 'text-red-600', dot: 'bg-red-500' },
  amber: { bg: 'bg-amber-50', border: 'border-amber-500', text: 'text-amber-600', dot: 'bg-amber-500' },
  emerald: { bg: 'bg-emerald-50', border: 'border-emerald-500', text: 'text-emerald-600', dot: 'bg-emerald-500' },
  blue: { bg: 'bg-blue-50', border: 'border-blue-500', text: 'text-blue-600', dot: 'bg-blue-500' },
  violet: { bg: 'bg-violet-50', border: 'border-violet-500', text: 'text-violet-600', dot: 'bg-violet-500' },
  orange: { bg: 'bg-orange-50', border: 'border-orange-500', text: 'text-orange-600', dot: 'bg-orange-500' },
};

function Arrow() {
  return (
    <div className="flex items-center justify-center w-8 shrink-0">
      <svg width="24" height="12" viewBox="0 0 24 12" className="text-gray-300">
        <path
          d="M0 6h18M14 1l6 5-6 5"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </svg>
    </div>
  );
}

function StageCard({ stage }: { stage: typeof stages[0] }) {
  const colors = colorMap[stage.color];
  return (
    <div
      className={`
        relative flex flex-col items-center justify-center
        px-5 py-3 rounded-xl border-2
        ${colors.bg} ${colors.border}
        min-w-[140px] transition-all duration-200
        hover:shadow-lg hover:-translate-y-0.5
      `}
    >
      {/* Stage number badge */}
      <div
        className={`
          absolute -top-2 -left-2 w-6 h-6 rounded-full
          ${colors.dot} text-white text-xs font-bold
          flex items-center justify-center shadow-sm
        `}
      >
        {stage.num}
      </div>
      <div className={`font-semibold text-sm ${colors.text}`}>{stage.name}</div>
      <div className="text-xs text-gray-500 mt-0.5">{stage.tech}</div>
    </div>
  );
}

function DownArrow() {
  return (
    <div className="flex justify-end pr-[70px]">
      <svg width="20" height="24" viewBox="0 0 20 24" className="text-gray-300">
        <path
          d="M10 0v18M5 14l5 6 5-6"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </svg>
    </div>
  );
}

export function PipelineFlowDiagram() {
  const firstRow = stages.slice(0, 3);
  const secondRow = stages.slice(3);

  return (
    <div className="w-full my-8">
      {/* Row 1: Input → Stages 1-3 */}
      <div className="flex items-center justify-center gap-1">
        {/* Input */}
        <div className="flex flex-col items-center px-4 py-3 rounded-lg bg-gray-50 border border-gray-300 border-dashed min-w-[80px]">
          <div className="text-sm font-medium text-gray-500">Input</div>
          <div className="text-xs text-gray-400">Raw Text</div>
        </div>

        <Arrow />

        {firstRow.map((stage, i) => (
          <div key={stage.num} className="flex items-center">
            <StageCard stage={stage} />
            {i < firstRow.length - 1 && <Arrow />}
          </div>
        ))}
      </div>

      {/* Connector arrow down */}
      <DownArrow />

      {/* Row 2: Stages 4-6 → Output */}
      <div className="flex items-center justify-center gap-1">
        {secondRow.map((stage, i) => (
          <div key={stage.num} className="flex items-center">
            <StageCard stage={stage} />
            {i < secondRow.length - 1 && <Arrow />}
          </div>
        ))}

        <Arrow />

        {/* Output */}
        <div className="flex flex-col items-center px-4 py-3 rounded-lg bg-gray-50 border border-gray-300 border-dashed min-w-[80px]">
          <div className="text-sm font-medium text-gray-500">Output</div>
          <div className="text-xs text-gray-400">Statements</div>
        </div>
      </div>
    </div>
  );
}

// Data flow - shows the transformation of data types through the pipeline
const dataSteps = [
  { label: 'Text', color: 'gray' },
  { label: 'Atomic Statements', color: 'red' },
  { label: 'Typed Triples', color: 'amber' },
  { label: 'Qualified Entities', color: 'emerald' },
  { label: 'Canonical Entities', color: 'blue' },
  { label: 'Labeled Statements', color: 'violet' },
  { label: 'Taxonomy Results', color: 'orange' },
];

const dataColorMap: Record<string, { border: string; text: string; bg: string }> = {
  gray: { border: 'border-gray-300', text: 'text-gray-600', bg: 'bg-white' },
  red: { border: 'border-red-400', text: 'text-red-700', bg: 'bg-red-50' },
  amber: { border: 'border-amber-400', text: 'text-amber-700', bg: 'bg-amber-50' },
  emerald: { border: 'border-emerald-400', text: 'text-emerald-700', bg: 'bg-emerald-50' },
  blue: { border: 'border-blue-400', text: 'text-blue-700', bg: 'bg-blue-50' },
  violet: { border: 'border-violet-400', text: 'text-violet-700', bg: 'bg-violet-50' },
  orange: { border: 'border-orange-400', text: 'text-orange-700', bg: 'bg-orange-50' },
};

export function DataFlowDiagram() {
  return (
    <div className="w-full my-8 overflow-x-auto">
      <div className="flex items-center justify-center min-w-max px-4">
        {dataSteps.map((step, i) => {
          const colors = dataColorMap[step.color];
          return (
            <div key={step.label} className="flex items-center">
              <div
                className={`
                  px-3 py-2 rounded-lg border-2 ${colors.border} ${colors.bg}
                  text-xs font-medium ${colors.text}
                  whitespace-nowrap
                `}
              >
                {step.label}
              </div>
              {i < dataSteps.length - 1 && (
                <div className="w-6 flex items-center justify-center">
                  <svg width="16" height="8" viewBox="0 0 16 8" className="text-gray-300">
                    <path
                      d="M0 4h12M10 1l3 3-3 3"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="1.5"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    />
                  </svg>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

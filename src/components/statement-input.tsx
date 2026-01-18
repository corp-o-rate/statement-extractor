'use client';

import { useState } from 'react';
import { Loader2, Sparkles, Trash2, Shuffle } from 'lucide-react';
import { CACHED_INPUT } from '@/lib/cached-example';
import { toast } from 'sonner';

interface StatementInputProps {
  onExtract: (text: string, options?: { useCanonicalPredicates?: boolean }) => void;
  isLoading: boolean;
  elapsedSeconds?: number;
}

const EXAMPLE_TEXTS = [
  {
    label: 'Corporate ESG',
    text: CACHED_INPUT,
  },
  {
    label: 'News Article',
    text: `Amazon announced plans to hire 150,000 workers for the holiday season. The company's CEO Andy Jassy said the positions would include warehouse and delivery roles across the United States. Labor unions criticized the move, citing concerns about working conditions. The announcement comes as retail giants prepare for increased consumer spending during Black Friday and Cyber Monday.`,
  },
  {
    label: 'Scientific',
    text: `Researchers at MIT have developed a new battery technology that could triple the range of electric vehicles. The team, led by Professor Jennifer Chen, published their findings in Nature Energy. Tesla has expressed interest in licensing the technology. The breakthrough uses solid-state lithium cells instead of traditional liquid electrolytes, significantly improving energy density.`,
  },
];

export function StatementInput({ onExtract, isLoading, elapsedSeconds = 0 }: StatementInputProps) {
  const [text, setText] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [useCanonicalPredicates, setUseCanonicalPredicates] = useState(false);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (text.trim() && !isLoading) {
      onExtract(text.trim(), { useCanonicalPredicates });
    }
  };

  const loadExample = (exampleText: string) => {
    setText(exampleText);
  };

  const generateRandomText = async () => {
    setIsGenerating(true);
    setText(''); // Clear existing text before streaming
    try {
      const response = await fetch('/api/generate', { method: 'POST' });
      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || 'Failed to generate text');
      }

      // Read the stream
      const reader = response.body?.getReader();
      if (!reader) {
        throw new Error('No response body');
      }

      const decoder = new TextDecoder();
      let streamedText = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });
        streamedText += chunk;
        setText(streamedText);
      }

      toast.success('Random text generated!');
    } catch (error) {
      console.error('Generation error:', error);
      toast.error(error instanceof Error ? error.message : 'Failed to generate text');
    } finally {
      setIsGenerating(false);
    }
  };

  const clearText = () => {
    setText('');
  };

  const charCount = text.length;
  const maxChars = 4000;

  return (
    <div className="w-full">
      <form onSubmit={handleSubmit}>
        {/* Example buttons */}
        <div className="mb-4 flex flex-wrap gap-2">
          <span className="text-sm text-gray-500 mr-2 self-center">Try an example:</span>
          {EXAMPLE_TEXTS.map((example) => (
            <button
              key={example.label}
              type="button"
              onClick={() => loadExample(example.text)}
              className="px-3 py-1.5 text-sm font-medium border border-gray-200 hover:border-black transition-colors"
              disabled={isLoading || isGenerating}
            >
              {example.label}
            </button>
          ))}
          <button
            type="button"
            onClick={generateRandomText}
            className="px-3 py-1.5 text-sm font-medium border border-gray-300 bg-gray-800 text-white hover:bg-gray-700 transition-colors inline-flex items-center gap-1.5"
            disabled={isLoading || isGenerating}
          >
            {isGenerating ? (
              <>
                <Loader2 className="w-3.5 h-3.5 spinner" />
                Generating...
              </>
            ) : (
              <>
                <Shuffle className="w-3.5 h-3.5" />
                Randomly Generated
              </>
            )}
          </button>
        </div>

        {/* Textarea */}
        <div className="relative">
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Paste your text here to extract statements..."
            className="input-textarea min-h-[200px]"
            maxLength={maxChars}
            disabled={isLoading}
          />

          {/* Character count */}
          <div className="absolute bottom-2 right-2 text-xs text-gray-400">
            {charCount.toLocaleString()} / {maxChars.toLocaleString()}
          </div>
        </div>

        {/* Options */}
        <div className="mt-4 flex items-center gap-3">
          <label className="inline-flex items-center gap-2 cursor-pointer select-none">
            <input
              type="checkbox"
              checked={useCanonicalPredicates}
              onChange={(e) => setUseCanonicalPredicates(e.target.checked)}
              disabled={isLoading}
              className="w-4 h-4 accent-red-600 cursor-pointer"
            />
            <span className="text-sm text-gray-700">
              Use canonical predicates
            </span>
            <a
              href="#canonical-predicates"
              className="text-xs text-gray-400 hover:text-red-600 transition-colors"
              title="View list of canonical predicates"
            >
              (view list)
            </a>
          </label>
        </div>

        {/* Actions */}
        <div className="mt-4 flex flex-wrap gap-3 items-center justify-between">
          <div className="flex gap-3">
            <button
              type="submit"
              disabled={!text.trim() || isLoading}
              className="btn-brutal"
            >
              {isLoading ? (
                <>
                  <Loader2 className="w-4 h-4 mr-2 spinner" />
                  {elapsedSeconds > 0 ? `Processing... (${elapsedSeconds}s)` : 'Submitting...'}
                </>
              ) : (
                <>
                  <Sparkles className="w-4 h-4 mr-2" />
                  Extract Statements
                </>
              )}
            </button>

            {text && (
              <button
                type="button"
                onClick={clearText}
                className="btn-secondary"
                disabled={isLoading}
              >
                <Trash2 className="w-4 h-4 mr-2" />
                Clear
              </button>
            )}
          </div>

          <p className="text-sm text-gray-500">
            Powered by{' '}
            <a
              href="https://corp-o-rate.com"
              target="_blank"
              rel="noopener noreferrer"
              className="text-red-600 hover:underline"
            >
              corp-o-rate
            </a>
          </p>
        </div>
      </form>
    </div>
  );
}

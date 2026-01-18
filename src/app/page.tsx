'use client';

import { useState, useEffect } from 'react';
import { Header } from '@/components/header';
import { Footer } from '@/components/footer';
import { StatementInput } from '@/components/statement-input';
import { StatementList } from '@/components/statement-list';
import { StatementEditor } from '@/components/statement-editor';
import { RelationshipGraph } from '@/components/relationship-graph';
import { RateLimitBanner } from '@/components/rate-limit-banner';
import { QuickStart } from '@/components/documentation';
import { LLMPrompts } from '@/components/llm-prompts';
import { ExtractionResult, Statement, JobSubmissionResponse, JobStatusResponse } from '@/lib/types';
import { getUserUuid } from '@/lib/user-uuid';
import { toast } from 'sonner';
import { ExportFormats } from '@/components/export-formats';
import { Network, FileText, BookOpen, Bot, Edit3, Eye, Code } from 'lucide-react';
import { HowItWorks, AboutCorpORate } from '@/components/about-sections';
import { CanonicalPredicates } from '@/components/canonical-predicates';
import { WarmUpDialog } from '@/components/warm-up-dialog';

// Polling interval in milliseconds
const POLL_INTERVAL = 5000;
// Show warm-up dialog after this many seconds
const WARMUP_DIALOG_THRESHOLD = 30;

export default function Home() {
  const [statements, setStatements] = useState<Statement[]>([]);
  const [editedStatements, setEditedStatements] = useState<Statement[]>([]);
  const [inputText, setInputText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isEditMode, setIsEditMode] = useState(false);
  const [hasChanges, setHasChanges] = useState(false);
  const [rateLimitMessage, setRateLimitMessage] = useState<string | undefined>();
  const [userUuid, setUserUuid] = useState('');
  const [isLiking, setIsLiking] = useState(false);
  const [hasLiked, setHasLiked] = useState(false);
  const [showWarmUpDialog, setShowWarmUpDialog] = useState(false);
  const [showTimeoutDialog, setShowTimeoutDialog] = useState(false);
  const [warmUpDialogDismissed, setWarmUpDialogDismissed] = useState(false);

  useEffect(() => {
    setUserUuid(getUserUuid());
  }, []);

  // Show warm-up dialog after threshold seconds of loading
  useEffect(() => {
    if (isLoading && elapsedSeconds >= WARMUP_DIALOG_THRESHOLD && !warmUpDialogDismissed) {
      setShowWarmUpDialog(true);
    }
  }, [isLoading, elapsedSeconds, warmUpDialogDismissed]);

  const pollJobStatus = async (
    jobId: string,
    inputTextForCache?: string,
    useCanonicalPredicates?: boolean
  ): Promise<JobStatusResponse> => {
    const params = new URLSearchParams({ jobId });
    if (inputTextForCache) {
      params.set('inputText', inputTextForCache);
    }
    if (useCanonicalPredicates) {
      params.set('useCanonicalPredicates', 'true');
    }
    const response = await fetch(`/api/extract/status?${params.toString()}`);
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || 'Failed to check job status');
    }
    return response.json();
  };

  const handleExtract = async (text: string, options?: { useCanonicalPredicates?: boolean }) => {
    setIsLoading(true);
    setElapsedSeconds(0);
    setRateLimitMessage(undefined);
    setInputText(text);
    setHasLiked(false); // Reset like status for new extraction
    setShowWarmUpDialog(false);
    setShowTimeoutDialog(false);
    setWarmUpDialogDismissed(false);

    // Start elapsed time counter
    const startTime = Date.now();
    const timerInterval = setInterval(() => {
      setElapsedSeconds(Math.floor((Date.now() - startTime) / 1000));
    }, 1000);

    try {
      const response = await fetch('/api/extract', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text,
          useCanonicalPredicates: options?.useCanonicalPredicates,
        }),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || 'Failed to extract statements');
      }

      const result = await response.json();

      // Check if this is a job submission (has jobId) or immediate result (has statements)
      if (result.jobId) {
        // Async job - poll for result
        const jobSubmission = result as JobSubmissionResponse;
        console.log(`Job submitted: ${jobSubmission.jobId}`);

        let statusResult: JobStatusResponse;
        do {
          await new Promise(resolve => setTimeout(resolve, POLL_INTERVAL));
          statusResult = await pollJobStatus(jobSubmission.jobId, text, options?.useCanonicalPredicates);
          console.log(`Job status: ${statusResult.status}`);
        } while (statusResult.status === 'IN_QUEUE' || statusResult.status === 'IN_PROGRESS');

        if (statusResult.status === 'FAILED') {
          throw new Error(statusResult.error || 'Job failed');
        }

        if (statusResult.status === 'TIMED_OUT') {
          // Show timeout dialog instead of error toast
          clearInterval(timerInterval);
          setIsLoading(false);
          setShowWarmUpDialog(false);
          setShowTimeoutDialog(true);
          return; // Exit early - don't cache, don't show error toast
        }

        // Job completed
        const statements = statusResult.statements || [];
        setStatements(statements);
        setEditedStatements(JSON.parse(JSON.stringify(statements)));
        setHasChanges(false);
        setIsEditMode(false);

        if (statements.length === 0) {
          toast.info('No statements found in the text');
        } else {
          toast.success(`Extracted ${statements.length} statement${statements.length !== 1 ? 's' : ''}`);
        }
      } else {
        // Immediate result (local model or cached)
        const extractionResult = result as ExtractionResult;
        setStatements(extractionResult.statements);
        setEditedStatements(JSON.parse(JSON.stringify(extractionResult.statements)));
        setHasChanges(false);
        setIsEditMode(false);

        if (extractionResult.cached && extractionResult.message) {
          setRateLimitMessage(extractionResult.message);
          toast.warning(extractionResult.message);
        } else if (extractionResult.statements.length === 0) {
          toast.info('No statements found in the text');
        } else {
          toast.success(`Extracted ${extractionResult.statements.length} statement${extractionResult.statements.length !== 1 ? 's' : ''}`);
        }
      }
    } catch (error) {
      console.error('Extraction error:', error);
      toast.error(error instanceof Error ? error.message : 'Failed to extract statements');
    } finally {
      clearInterval(timerInterval);
      setIsLoading(false);
      setElapsedSeconds(0);
      setShowWarmUpDialog(false);
    }
  };

  const handleStatementsChange = (newStatements: Statement[]) => {
    setEditedStatements(newStatements);
    setHasChanges(JSON.stringify(newStatements) !== JSON.stringify(statements));
  };

  const handleSubmitCorrection = async () => {
    if (!inputText) {
      toast.error('No input text to submit');
      return;
    }

    setIsSubmitting(true);

    try {
      const response = await fetch('/api/corrections', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          inputText,
          statements: editedStatements,
          userUuid,
          source: 'correction',
        }),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || 'Failed to submit correction');
      }

      toast.success('Correction submitted! Thank you for contributing.');
      setStatements(editedStatements);
      setHasChanges(false);
    } catch (error) {
      console.error('Submit error:', error);
      toast.error(error instanceof Error ? error.message : 'Failed to submit correction');
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleLike = async () => {
    if (!inputText || statements.length === 0) {
      return;
    }

    setIsLiking(true);

    try {
      const response = await fetch('/api/corrections', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          inputText,
          statements,
          userUuid,
          source: 'liked',
        }),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || 'Failed to save');
      }

      setHasLiked(true);
      toast.success('Thanks for the feedback!');
    } catch (error) {
      console.error('Like error:', error);
      toast.error(error instanceof Error ? error.message : 'Failed to save');
    } finally {
      setIsLiking(false);
    }
  };

  const toggleEditMode = () => {
    if (isEditMode) {
      // Switching to view mode - reset changes if any
      setEditedStatements(JSON.parse(JSON.stringify(statements)));
      setHasChanges(false);
    }
    setIsEditMode(!isEditMode);
  };

  return (
    <div className="min-h-screen flex flex-col">
      <Header />

      <main className="flex-1">
        {/* Hero Section */}
        <section className="py-12 px-4 sm:px-6 lg:px-8 border-b">
          <div className="max-w-6xl mx-auto">
            <div className="text-center mb-8">
              <span className="section-label">corp-extractor demo</span>
              <h1 className="text-4xl md:text-5xl font-black mt-4 tracking-tight">
                EXTRACT STATEMENTS.
                <br />
                <span className="text-gray-400">MAP RELATIONSHIPS.</span>
              </h1>
              <p className="mt-4 text-gray-600 max-w-2xl mx-auto">
                This demo showcases a python library which uses a custom fine-tuned{' '}
                <a
                  href="https://blog.google/technology/developers/t5gemma-2/"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-red-600 hover:underline font-medium"
                >
                  T5-Gemma 2
                </a>{' '}
                model that extracts atomic subject-predicate-object statements from unstructured text and resolves coreferences (e.g., pronouns to named entities),
                inspired by{' '}
                <a
                  href="https://www.microsoft.com/en-us/research/publication/towards-effective-extraction-and-evaluation-of-factual-claims/"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-red-600 hover:underline font-medium"
                >
                  Microsoft Research&apos;s work on factual claim extraction
                </a>.
                The library then identifies entities (organizations, people, locations, dates, etc. ) and their
                relationships using GLiNER2, and outputs structured data for downstream tasks like fact-checking,
                knowledge graph construction, and information retrieval.
              </p>
            </div>

            {/* Input Section */}
            <div className="brutal-card p-6 md:p-8">
              <StatementInput onExtract={handleExtract} isLoading={isLoading} elapsedSeconds={elapsedSeconds} />
            </div>
          </div>
        </section>

        {/* Results Section */}
        <section className="py-12 px-4 sm:px-6 lg:px-8">
          <div className="max-w-6xl mx-auto">
            <RateLimitBanner message={rateLimitMessage} />

            <div className="grid lg:grid-cols-2 gap-8">
              {/* Statements List / Editor */}
              <div>
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center gap-2">
                    <FileText className="w-5 h-5 text-red-600" />
                    <h2 className="font-bold text-xl">Statements</h2>
                  </div>
                  {statements.length > 0 && (
                    <button
                      onClick={toggleEditMode}
                      className="inline-flex items-center gap-2 px-3 py-1.5 text-sm font-semibold text-gray-600 hover:text-black border rounded transition-colors"
                    >
                      {isEditMode ? (
                        <>
                          <Eye className="w-4 h-4" />
                          View
                        </>
                      ) : (
                        <>
                          <Edit3 className="w-4 h-4" />
                          Correct
                        </>
                      )}
                    </button>
                  )}
                </div>
                <div className="editorial-card p-4 md:p-6 min-h-[400px]">
                  {isEditMode ? (
                    <StatementEditor
                      statements={editedStatements}
                      onChange={handleStatementsChange}
                      onSubmit={handleSubmitCorrection}
                      isSubmitting={isSubmitting}
                      hasChanges={hasChanges}
                    />
                  ) : (
                    <StatementList
                      statements={statements}
                      onLike={handleLike}
                      isLiking={isLiking}
                      hasLiked={hasLiked}
                    />
                  )}
                </div>
              </div>

              {/* Relationship Graph */}
              <div>
                <div className="flex items-center gap-2 mb-4">
                  <Network className="w-5 h-5 text-red-600" />
                  <h2 className="font-bold text-xl">Relationship Graph</h2>
                </div>
                <RelationshipGraph statements={isEditMode ? editedStatements : statements} />
              </div>
            </div>

            {/* Export Formats */}
            <div className="mt-8">
              <div className="flex items-center gap-2 mb-4">
                <Code className="w-5 h-5 text-red-600" />
                <h2 className="font-bold text-xl">Export</h2>
              </div>
              <ExportFormats statements={isEditMode ? editedStatements : statements} />
            </div>

            {/* Canonical Predicates Reference */}
            <div className="mt-8">
              <CanonicalPredicates />
            </div>
          </div>
        </section>

        {/* Quick Start Section */}
        <section className="py-12 px-4 sm:px-6 lg:px-8 bg-gray-50/50">
          <div className="max-w-6xl mx-auto">
            <div className="flex items-center gap-2 mb-6">
              <BookOpen className="w-5 h-5 text-red-600" />
              <h2 className="font-bold text-xl">Quick Start</h2>
            </div>
            <QuickStart />
          </div>
        </section>

        {/* LLM Prompts Section */}
        <section className="py-12 px-4 sm:px-6 lg:px-8">
          <div className="max-w-6xl mx-auto">
            <div className="flex items-center gap-2 mb-6">
              <Bot className="w-5 h-5 text-red-600" />
              <h2 className="font-bold text-xl">For AI Assistants</h2>
            </div>
            <LLMPrompts />
          </div>
        </section>

        {/* About Section */}
        <section className="py-16 px-4 sm:px-6 lg:px-8 bg-gray-50 border-t">
          <div className="max-w-4xl mx-auto text-center">
            <span className="section-label">ABOUT THE MODEL</span>
            <h2 className="text-2xl md:text-3xl font-black mt-4">
              T5-Gemma 2 Statement Extractor
            </h2>
            <div className="mt-6 text-gray-600 space-y-4 text-left max-w-2xl mx-auto">
              <p>
                This model is based on Google&apos;s T5-Gemma 2 architecture (540M parameters) and has been
                fine-tuned on 77,515 examples of statement extraction from corporate and news documents.
              </p>
              <p>
                <strong>Key capabilities:</strong>
              </p>
              <ul className="list-disc list-inside space-y-1 ml-4">
                <li>Extract subject-predicate-object triples from text</li>
                <li>Identify entity types (ORG, PERSON, GPE, EVENT, etc.)</li>
                <li>Resolve coreferences (pronouns → entity names)</li>
                <li>Generate full resolved statement text</li>
              </ul>
              <p>
                <strong>Training details:</strong> Final eval loss of 0.209, trained with beam search
                (num_beams=4) for high-quality outputs.
              </p>
            </div>
          </div>
        </section>

        {/* How It Works Section */}
        <HowItWorks />

        {/* About Corp-o-Rate Section */}
        <AboutCorpORate />
      </main>

      <Footer />

      {/* Warm-up Dialog - shown after 30 seconds of loading */}
      <WarmUpDialog
        isOpen={showWarmUpDialog}
        elapsedSeconds={elapsedSeconds}
        onClose={() => {
          setShowWarmUpDialog(false);
          setWarmUpDialogDismissed(true);
        }}
      />

      {/* Timeout Dialog - shown when request times out */}
      <WarmUpDialog
        isOpen={showTimeoutDialog}
        elapsedSeconds={elapsedSeconds}
        onClose={() => setShowTimeoutDialog(false)}
        isTimeout
      />
    </div>
  );
}

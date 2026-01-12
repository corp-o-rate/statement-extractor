'use client';

import { useState, useEffect } from 'react';
import { Header } from '@/components/header';
import { Footer } from '@/components/footer';
import { StatementInput } from '@/components/statement-input';
import { StatementList } from '@/components/statement-list';
import { StatementEditor } from '@/components/statement-editor';
import { RelationshipGraph } from '@/components/relationship-graph';
import { RateLimitBanner } from '@/components/rate-limit-banner';
import { Documentation } from '@/components/documentation';
import { LLMPrompts } from '@/components/llm-prompts';
import { ExtractionResult, Statement, JobSubmissionResponse, JobStatusResponse } from '@/lib/types';
import { getUserUuid } from '@/lib/user-uuid';
import { toast } from 'sonner';
import { Network, FileText, BookOpen, Bot, Edit3, Eye } from 'lucide-react';

// Polling interval in milliseconds
const POLL_INTERVAL = 5000;

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

  useEffect(() => {
    setUserUuid(getUserUuid());
  }, []);

  const pollJobStatus = async (jobId: string): Promise<JobStatusResponse> => {
    const response = await fetch(`/api/extract/status?jobId=${jobId}`);
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || 'Failed to check job status');
    }
    return response.json();
  };

  const handleExtract = async (text: string) => {
    setIsLoading(true);
    setElapsedSeconds(0);
    setRateLimitMessage(undefined);
    setInputText(text);

    // Start elapsed time counter
    const startTime = Date.now();
    const timerInterval = setInterval(() => {
      setElapsedSeconds(Math.floor((Date.now() - startTime) / 1000));
    }, 1000);

    try {
      const response = await fetch('/api/extract', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
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
          statusResult = await pollJobStatus(jobSubmission.jobId);
          console.log(`Job status: ${statusResult.status}`);
        } while (statusResult.status === 'IN_QUEUE' || statusResult.status === 'IN_PROGRESS');

        if (statusResult.status === 'FAILED') {
          throw new Error(statusResult.error || 'Job failed');
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
              <span className="section-label">NLP MODEL DEMO</span>
              <h1 className="text-4xl md:text-5xl font-black mt-4 tracking-tight">
                EXTRACT STATEMENTS.
                <br />
                <span className="text-gray-400">VISUALIZE RELATIONSHIPS.</span>
              </h1>
              <p className="mt-4 text-gray-600 max-w-2xl mx-auto">
                Transform unstructured text into structured statements using our T5-Gemma 2 model.
                Identify subjects, objects, predicates, and entity types automatically.
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
                    <StatementList statements={statements} />
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
          </div>
        </section>

        {/* Documentation Section */}
        <section className="py-12 px-4 sm:px-6 lg:px-8 bg-gray-50/50">
          <div className="max-w-6xl mx-auto">
            <div className="flex items-center gap-2 mb-6">
              <BookOpen className="w-5 h-5 text-red-600" />
              <h2 className="font-bold text-xl">Documentation</h2>
            </div>
            <Documentation />
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
      </main>

      <Footer />
    </div>
  );
}

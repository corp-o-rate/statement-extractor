import { NextRequest, NextResponse } from 'next/server';
import { parseStatements } from '@/lib/statement-parser';
import { setCachedStatements } from '@/lib/cache';

const RUNPOD_ENDPOINT_ID = process.env.RUNPOD_ENDPOINT_ID;
const RUNPOD_API_KEY = process.env.RUNPOD_API_KEY;

export async function GET(request: NextRequest) {
  const jobId = request.nextUrl.searchParams.get('jobId');
  const inputText = request.nextUrl.searchParams.get('inputText');
  const useCanonicalPredicates = request.nextUrl.searchParams.get('useCanonicalPredicates') === 'true';

  if (!jobId) {
    return NextResponse.json(
      { error: 'Missing jobId parameter' },
      { status: 400 }
    );
  }

  if (!RUNPOD_ENDPOINT_ID || !RUNPOD_API_KEY) {
    return NextResponse.json(
      { error: 'RunPod not configured' },
      { status: 500 }
    );
  }

  try {
    console.log(`Checking status for job: ${jobId}`);

    const response = await fetch(
      `https://api.runpod.ai/v2/${RUNPOD_ENDPOINT_ID}/status/${jobId}`,
      {
        headers: {
          'Authorization': `Bearer ${RUNPOD_API_KEY}`,
        },
      }
    );

    if (!response.ok) {
      const errorText = await response.text();
      console.error(`RunPod status error: status=${response.status}, body=${errorText}`);
      return NextResponse.json(
        { error: `Failed to check status: ${response.status}` },
        { status: response.status }
      );
    }

    const data = await response.json();
    console.log(`Job ${jobId} status: ${data.status}`);

    // Handle completed job - parse the output
    if (data.status === 'COMPLETED' && data.output) {
      // Check if this is a URL job result (has metadata field)
      if (data.output.metadata && data.output.statements) {
        // URL job result - statements are already structured
        const urlStatements = data.output.statements.map((stmt: {
          subject?: { text?: string; type?: string };
          object?: { text?: string; type?: string };
          predicate?: string;
          text?: string;
          labels?: Record<string, string>;
          taxonomy?: Array<{ taxonomy: string; category: string; label: string; confidence: number }>;
        }) => ({
          subject: {
            name: stmt.subject?.text || '',
            type: stmt.subject?.type || 'UNKNOWN',
          },
          object: {
            name: stmt.object?.text || '',
            type: stmt.object?.type || 'UNKNOWN',
          },
          predicate: stmt.predicate || '',
          text: stmt.text || '',
          labels: stmt.labels ? Object.entries(stmt.labels).map(([k, v]) => ({
            label_type: k,
            label_value: v,
            confidence: 1.0,
          })) : undefined,
          taxonomyResults: stmt.taxonomy?.map(t => ({
            taxonomy_name: t.taxonomy,
            category: t.category,
            label: t.label,
            confidence: t.confidence,
          })),
        }));

        return NextResponse.json({
          status: 'COMPLETED',
          statements: urlStatements,
          metadata: data.output.metadata,
          summary: data.output.summary,
          cached: data.output.cached || false,
        });
      }

      // Regular text extraction result
      // Check if response is in new JSON format (v0.2.0+)
      const outputData = data.output.output || data.output;
      const isJsonFormat = data.output.format === 'json' || (typeof outputData === 'object' && outputData.statements);

      let statements;
      if (isJsonFormat) {
        // New JSON format - already parsed or needs parsing
        statements = parseStatements(outputData);
      } else {
        // Legacy XML format
        statements = parseStatements(outputData);
      }

      // Cache the result if we have the input text
      // Note: setCachedStatements will skip empty results to prevent caching failures/timeouts
      if (inputText && statements.length > 0) {
        await setCachedStatements(inputText, statements, { useCanonicalPredicates });
      }

      return NextResponse.json({
        status: 'COMPLETED',
        statements,
        cached: false,
      });
    }

    // Handle failed job
    if (data.status === 'FAILED') {
      console.error(`Job ${jobId} failed:`, data.error);
      // DO NOT cache failed results
      return NextResponse.json({
        status: 'FAILED',
        error: data.error || 'Job failed',
      });
    }

    // Handle timed out job - DO NOT cache
    if (data.status === 'TIMED_OUT') {
      console.error(`Job ${jobId} timed out`);
      return NextResponse.json({
        status: 'TIMED_OUT',
        error: 'Request timed out. The server may be busy or starting up. Please try again.',
      });
    }

    // Job still in progress
    return NextResponse.json({
      status: data.status, // IN_QUEUE or IN_PROGRESS
    });

  } catch (error) {
    console.error('Status check error:', error);
    return NextResponse.json(
      { error: 'Failed to check job status' },
      { status: 500 }
    );
  }
}

import { NextRequest, NextResponse } from 'next/server';
import { parseStatements } from '@/lib/statement-parser';
import { CACHED_EXAMPLE } from '@/lib/cached-example';
import { ExtractionResult } from '@/lib/types';

// Environment configuration
const RUNPOD_ENDPOINT_ID = process.env.RUNPOD_ENDPOINT_ID;
const RUNPOD_API_KEY = process.env.RUNPOD_API_KEY;
const LOCAL_MODEL_URL = process.env.LOCAL_MODEL_URL;

// Timeout for RunPod requests (180 seconds for cold starts)
const RUNPOD_TIMEOUT = 180000;

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { text } = body;

    if (!text || typeof text !== 'string') {
      return NextResponse.json(
        { error: 'Missing or invalid text field' },
        { status: 400 }
      );
    }

    if (text.length > 10000) {
      return NextResponse.json(
        { error: 'Text too long. Maximum 10,000 characters.' },
        { status: 400 }
      );
    }

    // Wrap text in page tags as expected by the model
    const modelInput = `<page>${text}</page>`;

    // Try RunPod first (primary production option)
    if (RUNPOD_ENDPOINT_ID && RUNPOD_API_KEY) {
      try {
        console.log(`Calling RunPod endpoint: ${RUNPOD_ENDPOINT_ID}`);

        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), RUNPOD_TIMEOUT);

        const runpodResponse = await fetch(
          `https://api.runpod.ai/v2/${RUNPOD_ENDPOINT_ID}/runsync`,
          {
            method: 'POST',
            headers: {
              'Authorization': `Bearer ${RUNPOD_API_KEY}`,
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({
              input: { text: modelInput },
            }),
            signal: controller.signal,
          }
        );

        clearTimeout(timeoutId);

        if (runpodResponse.ok) {
          const data = await runpodResponse.json();

          if (data.status === 'COMPLETED' && data.output) {
            const outputText = data.output.output || data.output;
            const statements = parseStatements(outputText);

            const result: ExtractionResult = {
              statements,
              cached: false,
              inputText: text,
            };

            return NextResponse.json(result);
          }

          // Handle RunPod errors
          if (data.status === 'FAILED') {
            console.error('RunPod job failed:', data.error);
            throw new Error(data.error || 'RunPod job failed');
          }

          // Handle timeout/in-progress (shouldn't happen with runsync but just in case)
          if (data.status === 'IN_PROGRESS' || data.status === 'IN_QUEUE') {
            console.warn('RunPod job still processing, returning cached example');
            return NextResponse.json({
              ...CACHED_EXAMPLE,
              message: 'Model is processing. Showing cached example. Try again shortly.',
            });
          }
        } else {
          const errorText = await runpodResponse.text();
          console.error(`RunPod API error: status=${runpodResponse.status}, body=${errorText}`);
          throw new Error(`RunPod API error: ${runpodResponse.status}`);
        }
      } catch (runpodError) {
        if (runpodError instanceof Error && runpodError.name === 'AbortError') {
          console.warn('RunPod request timed out');
        } else {
          console.warn('RunPod unavailable:', runpodError);
        }
        // Fall through to next option
      }
    }

    // Try local model if configured
    if (LOCAL_MODEL_URL) {
      try {
        console.log(`Calling local model: ${LOCAL_MODEL_URL}`);

        const localResponse = await fetch(`${LOCAL_MODEL_URL}/extract`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text: modelInput }),
        });

        if (localResponse.ok) {
          const data = await localResponse.json();
          const statements = parseStatements(data.output || data.result);

          const result: ExtractionResult = {
            statements,
            cached: false,
            inputText: text,
          };

          return NextResponse.json(result);
        }
      } catch (localError) {
        console.warn('Local model unavailable:', localError);
      }
    }

    // No model available, return cached example
    console.log('No model endpoint configured, returning cached example');
    return NextResponse.json({
      ...CACHED_EXAMPLE,
      message: 'No model endpoint configured. Showing cached example. See documentation to run locally or deploy to RunPod.',
    });

  } catch (error) {
    console.error('Extract API error:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}

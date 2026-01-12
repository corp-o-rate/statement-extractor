import { NextRequest, NextResponse } from 'next/server';
import { parseStatements } from '@/lib/statement-parser';
import { CACHED_EXAMPLE } from '@/lib/cached-example';
import { ExtractionResult } from '@/lib/types';

const HF_MODEL = process.env.HF_MODEL || 'Corp-o-Rate-Community/statement-extractor';
const HF_TOKEN = process.env.HF_TOKEN;
const LOCAL_MODEL_URL = process.env.LOCAL_MODEL_URL;

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

    // Try local model first if configured
    if (LOCAL_MODEL_URL) {
      try {
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
        console.warn('Local model unavailable, falling back to HuggingFace:', localError);
      }
    }

    // Try HuggingFace Inference API
    if (!HF_TOKEN) {
      console.warn('No HF_TOKEN configured, returning cached example');
      return NextResponse.json({
        ...CACHED_EXAMPLE,
        message: 'API not configured. Showing cached example. See documentation to run locally.',
      });
    }

    try {
      const hfResponse = await fetch(
        `https://api-inference.huggingface.co/models/${HF_MODEL}`,
        {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${HF_TOKEN}`,
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            inputs: modelInput,
            parameters: {
              max_new_tokens: 2048,
              num_beams: 4,
              do_sample: false,
            },
          }),
        }
      );

      // Handle rate limits
      if (hfResponse.status === 429 || hfResponse.status === 503) {
        console.warn('HuggingFace rate limit hit, returning cached example');
        return NextResponse.json({
          ...CACHED_EXAMPLE,
          message: 'API rate limit reached. Showing cached example. Consider running locally for unlimited usage.',
        });
      }

      if (!hfResponse.ok) {
        const errorText = await hfResponse.text();
        console.error('HuggingFace API error:', hfResponse.status, errorText);

        // Model might be loading
        if (hfResponse.status === 503 || errorText.includes('loading')) {
          return NextResponse.json({
            ...CACHED_EXAMPLE,
            message: 'Model is loading. Showing cached example. Please try again in a few minutes.',
          });
        }

        return NextResponse.json({
          ...CACHED_EXAMPLE,
          message: 'API error. Showing cached example.',
        });
      }

      const data = await hfResponse.json();

      // HuggingFace returns different formats depending on the model type
      let outputText = '';
      if (Array.isArray(data)) {
        outputText = data[0]?.generated_text || data[0]?.text || '';
      } else if (typeof data === 'object') {
        outputText = data.generated_text || data[0]?.generated_text || '';
      } else if (typeof data === 'string') {
        outputText = data;
      }

      const statements = parseStatements(outputText);

      const result: ExtractionResult = {
        statements,
        cached: false,
        inputText: text,
      };

      return NextResponse.json(result);
    } catch (hfError) {
      console.error('HuggingFace API error:', hfError);
      return NextResponse.json({
        ...CACHED_EXAMPLE,
        message: 'API temporarily unavailable. Showing cached example.',
      });
    }
  } catch (error) {
    console.error('Extract API error:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}

import { NextRequest, NextResponse } from 'next/server';
import { createClient } from '@supabase/supabase-js';
import { Statement } from '@/lib/types';

const SUPABASE_URL = process.env.SUPABASE_URL!;
const SUPABASE_SERVICE_KEY = process.env.SUPABASE_SERVICE_KEY!;

/**
 * Convert statements array to XML format for training data.
 */
function statementsToXml(statements: Statement[]): string {
  const stmts = statements.map(stmt => {
    const subjectAttr = stmt.subject.type ? ` type="${stmt.subject.type}"` : '';
    const objectAttr = stmt.object.type ? ` type="${stmt.object.type}"` : '';

    return `<stmt><subject${subjectAttr}>${stmt.subject.name}</subject><object${objectAttr}>${stmt.object.name}</object><predicate>${stmt.predicate}</predicate><text>${stmt.text}</text></stmt>`;
  }).join('');

  return `<statements>${stmts}</statements>`;
}

export async function POST(request: NextRequest) {
  try {
    const { inputText, statements, userUuid } = await request.json();

    if (!inputText || !statements) {
      return NextResponse.json(
        { error: 'Missing inputText or statements' },
        { status: 400 }
      );
    }

    const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_KEY);

    const { error } = await supabase.from('statement_extractor_training').insert({
      input_text: `<page>${inputText}</page>`,
      output_xml: statementsToXml(statements),
      num_statements: statements.length,
      accepted: null, // Not yet accepted - needs review
      user_uuid: userUuid || null,
    });

    if (error) {
      console.error('Supabase error:', error);
      return NextResponse.json(
        { error: 'Failed to save correction' },
        { status: 500 }
      );
    }

    return NextResponse.json({ success: true });
  } catch (error) {
    console.error('Corrections API error:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}

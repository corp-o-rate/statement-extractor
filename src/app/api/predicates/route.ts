import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';

// Load predicates from the Python library's default_predicates.json
const PREDICATES_PATH = path.join(
  process.cwd(),
  'statement-extractor-lib/src/statement_extractor/data/default_predicates.json'
);

export async function GET() {
  try {
    const fileContents = fs.readFileSync(PREDICATES_PATH, 'utf-8');
    const predicates = JSON.parse(fileContents);

    return NextResponse.json(predicates);
  } catch (error) {
    console.error('Failed to load predicates:', error);
    return NextResponse.json(
      { error: 'Failed to load predicates' },
      { status: 500 }
    );
  }
}
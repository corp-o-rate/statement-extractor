import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';

// Load taxonomy from the Python library's statement_taxonomy.json
const TAXONOMY_PATH = path.join(
  process.cwd(),
  'statement-extractor-lib/src/statement_extractor/data/statement_taxonomy.json'
);

export async function GET() {
  try {
    const fileContents = fs.readFileSync(TAXONOMY_PATH, 'utf-8');
    const taxonomy = JSON.parse(fileContents);

    return NextResponse.json(taxonomy);
  } catch (error) {
    console.error('Failed to load taxonomy:', error);
    return NextResponse.json(
      { error: 'Failed to load taxonomy' },
      { status: 500 }
    );
  }
}
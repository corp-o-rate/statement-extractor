import { generateText } from 'ai';
import { openai } from '@ai-sdk/openai';
import { NextResponse } from 'next/server';

const TOPICS = [
  'a technology company announcing a new product or partnership',
  'a government policy change affecting businesses or citizens',
  'a scientific discovery with real-world applications',
  'an environmental initiative by a major corporation',
  'a merger or acquisition between two companies',
  'a sports team or athlete achieving a milestone',
  'a cultural event or festival and its economic impact',
  'a healthcare breakthrough or public health announcement',
  'an infrastructure project in a major city',
  'a financial institution reporting quarterly results',
  'an education reform or university research finding',
  'a legal case with implications for an industry',
];

export async function POST() {
  const apiKey = process.env.OPENAI_API_KEY;

  if (!apiKey) {
    return NextResponse.json(
      { error: 'OpenAI API key not configured' },
      { status: 503 }
    );
  }

  const topic = TOPICS[Math.floor(Math.random() * TOPICS.length)];

  try {
    const { text } = await generateText({
      model: openai('gpt-5-nano'),
      prompt: `Write a realistic news article or press release about ${topic}.

Requirements:
- Write exactly 3 paragraphs
- Minimum 250 words total
- Include specific names of people, organizations, and places
- Include dates, numbers, and statistics where appropriate
- Write in a professional journalistic style
- Make it factual-sounding (but fictional) with clear subject-predicate-object relationships
- Do not include a headline or title, just the body text`,
    });

    return NextResponse.json({ text: text.trim() });
  } catch (error) {
    console.error('Text generation error:', error);
    return NextResponse.json(
      { error: 'Failed to generate text' },
      { status: 500 }
    );
  }
}

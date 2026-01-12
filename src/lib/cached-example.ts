/**
 * Pre-cached example for rate limit fallback
 * This example is shown when the HuggingFace API rate limit is reached
 */

import { ExtractionResult } from './types';

export const CACHED_INPUT = `Apple Inc. announced today that it has committed to becoming carbon neutral across its entire supply chain by 2030. The tech giant's CEO, Tim Cook, stated that the company will invest $4.7 billion in renewable energy projects. Environmental groups, including Greenpeace, praised the announcement but called for faster action. The commitment comes amid growing pressure from shareholders and consumers for corporations to address climate change. Apple's sustainability report also revealed that the company has reduced its emissions by 40% since 2015.`;

export const CACHED_EXAMPLE: ExtractionResult = {
  statements: [
    {
      subject: { name: 'Apple Inc.', type: 'ORG' },
      object: { name: 'carbon neutral across its entire supply chain by 2030', type: 'EVENT' },
      predicate: 'committed to',
      text: 'Apple Inc. committed to becoming carbon neutral across its entire supply chain by 2030.',
    },
    {
      subject: { name: 'Tim Cook', type: 'PERSON' },
      object: { name: '$4.7 billion in renewable energy projects', type: 'MONEY' },
      predicate: 'stated investment of',
      text: 'Tim Cook stated that Apple will invest $4.7 billion in renewable energy projects.',
    },
    {
      subject: { name: 'Greenpeace', type: 'ORG' },
      object: { name: 'the announcement', type: 'EVENT' },
      predicate: 'praised',
      text: 'Greenpeace praised the announcement but called for faster action.',
    },
    {
      subject: { name: 'shareholders and consumers', type: 'PERSON' },
      object: { name: 'corporations', type: 'ORG' },
      predicate: 'pressuring to address climate change',
      text: 'Shareholders and consumers are pressuring corporations to address climate change.',
    },
    {
      subject: { name: 'Apple', type: 'ORG' },
      object: { name: 'emissions by 40% since 2015', type: 'PERCENT' },
      predicate: 'reduced',
      text: 'Apple has reduced its emissions by 40% since 2015.',
    },
  ],
  cached: true,
  message: 'API rate limit reached. Showing cached example. Run locally for unlimited usage.',
  inputText: CACHED_INPUT,
};

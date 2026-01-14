import { createClient, SupabaseClient } from '@supabase/supabase-js';
import { Statement } from './types';
import crypto from 'crypto';

const SUPABASE_URL = process.env.SUPABASE_URL;
const SUPABASE_SERVICE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY;

const CACHE_TTL_HOURS = 24;

function getSupabaseClient(): SupabaseClient | null {
  if (!SUPABASE_URL || !SUPABASE_SERVICE_KEY) {
    return null;
  }
  return createClient(SUPABASE_URL, SUPABASE_SERVICE_KEY);
}

interface CacheOptions {
  useCanonicalPredicates?: boolean;
}

function hashInput(text: string, options?: CacheOptions): string {
  // Include options in hash so different settings are cached separately
  const cacheKey = `${text}:canonical=${options?.useCanonicalPredicates || false}`;
  return crypto.createHash('sha256').update(cacheKey).digest('hex');
}

/**
 * Get cached statements for input text if exists and not expired.
 * Returns null if not found or expired.
 */
export async function getCachedStatements(
  inputText: string,
  options?: CacheOptions
): Promise<Statement[] | null> {
  const supabase = getSupabaseClient();
  if (!supabase) {
    return null;
  }

  const inputHash = hashInput(inputText, options);
  const cutoffTime = new Date(Date.now() - CACHE_TTL_HOURS * 60 * 60 * 1000).toISOString();

  try {
    const { data, error } = await supabase
      .from('statement_extractor_cache')
      .select('output_statements, created_at')
      .eq('input_hash', inputHash)
      .gte('created_at', cutoffTime)
      .order('created_at', { ascending: false })
      .limit(1)
      .single();

    if (error || !data) {
      return null;
    }

    console.log(`Cache hit for input hash: ${inputHash.substring(0, 8)}...`);
    return data.output_statements as Statement[];
  } catch (error) {
    console.warn('Cache lookup error:', error);
    return null;
  }
}

/**
 * Store statements in cache for input text.
 */
export async function setCachedStatements(
  inputText: string,
  statements: Statement[],
  options?: CacheOptions
): Promise<void> {
  const supabase = getSupabaseClient();
  if (!supabase) {
    return;
  }

  const inputHash = hashInput(inputText, options);

  try {
    const { error } = await supabase
      .from('statement_extractor_cache')
      .upsert(
        {
          input_hash: inputHash,
          input_text: inputText,
          output_statements: statements,
          created_at: new Date().toISOString(),
        },
        { onConflict: 'input_hash' }
      );

    if (error) {
      console.warn('Cache write error:', error);
    } else {
      console.log(`Cached result for input hash: ${inputHash.substring(0, 8)}...`);
    }
  } catch (error) {
    console.warn('Cache write error:', error);
  }
}

/**
 * Clean up expired cache entries (optional, can be run periodically).
 */
export async function cleanExpiredCache(): Promise<number> {
  const supabase = getSupabaseClient();
  if (!supabase) {
    return 0;
  }

  const cutoffTime = new Date(Date.now() - CACHE_TTL_HOURS * 60 * 60 * 1000).toISOString();

  try {
    const { data, error } = await supabase
      .from('statement_extractor_cache')
      .delete()
      .lt('created_at', cutoffTime)
      .select('input_hash');

    if (error) {
      console.warn('Cache cleanup error:', error);
      return 0;
    }

    return data?.length || 0;
  } catch (error) {
    console.warn('Cache cleanup error:', error);
    return 0;
  }
}

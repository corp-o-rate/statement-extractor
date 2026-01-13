-- Migration: Create statement_extractor_training table
-- Run this in Supabase SQL Editor

CREATE TABLE IF NOT EXISTS statement_extractor_training (
  id SERIAL PRIMARY KEY,
  input_text TEXT NOT NULL,
  output_xml TEXT NOT NULL,
  num_statements INTEGER NOT NULL,
  accepted BOOLEAN DEFAULT NULL,
  user_uuid UUID,
  source VARCHAR(20) DEFAULT 'upload',  -- 'upload', 'correction', 'liked'
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index for filtering
CREATE INDEX IF NOT EXISTS idx_training_accepted ON statement_extractor_training(accepted);
CREATE INDEX IF NOT EXISTS idx_training_user ON statement_extractor_training(user_uuid);
CREATE INDEX IF NOT EXISTS idx_training_source ON statement_extractor_training(source);

-- Migration to add source column to existing table:
ALTER TABLE statement_extractor_training ADD COLUMN IF NOT EXISTS source VARCHAR(20) DEFAULT 'upload';

-- Unlogged table for faster writes (data may be lost on crash, fine for cache)
CREATE UNLOGGED TABLE statement_extractor_cache (
                                                    input_hash TEXT PRIMARY KEY,
                                                    input_text TEXT NOT NULL,
                                                    output_statements JSONB NOT NULL,
                                                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Index for cleanup queries
CREATE INDEX idx_cache_created_at ON statement_extractor_cache (created_at);

-- Optional: Enable RLS but allow service role full access
ALTER TABLE statement_extractor_cache ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Service role has full access" ON statement_extractor_cache
    FOR ALL
    USING (true)
    WITH CHECK (true);

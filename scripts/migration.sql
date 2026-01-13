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

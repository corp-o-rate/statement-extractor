"""
Upload existing training data to Supabase.

Usage:
    uv run python upload_training_data.py

Requires .env file with:
    SUPABASE_URL=https://xxx.supabase.co
    SUPABASE_SERVICE_KEY=eyJ...
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_KEY"]

# Training data file
TRAINING_DATA_PATH = Path.home() / "tmp/train_30_dec_2025/training_data/coref_training.jsonl"

def main():
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

    if not TRAINING_DATA_PATH.exists():
        print(f"Training data not found: {TRAINING_DATA_PATH}")
        return

    print(f"Loading training data from {TRAINING_DATA_PATH}")

    records = []
    with open(TRAINING_DATA_PATH) as f:
        for line in f:
            data = json.loads(line)
            records.append({
                "input_text": data["input"],
                "output_xml": data["output"],
                "num_statements": data.get("num_statements", 0),
                "accepted": True,  # Existing data is accepted
                "user_uuid": None,  # No user for existing data
            })

    print(f"Loaded {len(records)} records")

    # Insert in batches of 100
    batch_size = 100
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        result = supabase.table("statement_extractor_training").insert(batch).execute()
        print(f"Inserted batch {i // batch_size + 1} ({len(batch)} records)")

    print("Done!")


if __name__ == "__main__":
    main()

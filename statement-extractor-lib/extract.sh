#!/bin/bash
# Simple wrapper script for corp-extractor CLI
#
# Usage:
#   ./extract.sh "Your text here"
#   ./extract.sh -f input.txt
#   cat input.txt | ./extract.sh -
#
# Requires: pip install corp-extractor[embeddings]

set -e

# Check if corp-extractor is installed
if ! command -v corp-extractor &> /dev/null; then
    echo "Error: corp-extractor not found. Install with:"
    echo "  pip install corp-extractor[embeddings]"
    exit 1
fi

# Pass all arguments to corp-extractor
corp-extractor "$@"

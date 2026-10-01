#!/bin/bash
# backend/run.sh
# Run pipeline with backend venv

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate venv
source venv311/bin/activate

# Run the pipeline
python -m app.preprocessing.pipeline "$@"
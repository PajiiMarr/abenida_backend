#!/bin/bash
# backend/activate.sh
# Activate backend virtual environment

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -d "venv311" ]; then
    source venv311/bin/activate
    echo "✅ Backend virtual environment activated"
    echo "📦 Python: $(which python)"
    echo "📍 Location: $SCRIPT_DIR"
else
    echo "❌ Virtual environment not found."
    echo "Run: python3 -m venv venv311"
    echo "Then: pip install -r requirements.txt"
fi
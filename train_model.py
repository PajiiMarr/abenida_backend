#!/usr/bin/env python
"""
Train flood detection model
Run from backend/ directory:
    python train_model.py
"""

import sys
from pathlib import Path

# Add app to path
sys.path.insert(0, str(Path(__file__).parent / 'app'))

from app.training.trainer import main

if __name__ == "__main__":
    main()
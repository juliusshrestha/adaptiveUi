#!/usr/bin/env python3
"""
Main entry point for Adaptive UI System
Run this script from the project root directory
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Now import and run the main system
from src.main import main

if __name__ == "__main__":
    main()


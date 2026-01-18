#!/bin/bash
# Setup script for Adaptive UI Research Project
# Creates virtual environment and installs dependencies

set -e

echo "Setting up Adaptive UI Research Project environment..."

# Soft guard: Python wheels for OpenCV/MediaPipe are not consistently available on Python 3.13 yet.
PY_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "unknown")
if [[ "$PY_VER" == "3.13" ]]; then
  echo ""
  echo "WARNING: Detected Python $PY_VER."
  echo "OpenCV/MediaPipe may fail to install on Python 3.13."
  echo "Recommended: use conda + Python 3.11 (see environment.yml)."
  echo ""
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements/base.txt

echo ""
echo "Setup complete!"
echo ""
echo "To activate the environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To deactivate, run:"
echo "  deactivate"

echo ""
echo "Optional: install full (research/dev) dependency set:"
echo "  pip install -r requirements.txt"


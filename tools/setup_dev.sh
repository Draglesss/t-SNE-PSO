#!/bin/bash
# Script to set up development environment
# Usage: bash tools/setup_dev.sh

set -e  # Exit on error

# Ensure we're at the repo root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "=== Setting up development environment ==="

# Install package in development mode
python -m pip install -e .

# Install development dependencies
python -m pip install --upgrade pip
python -m pip install pytest pytest-cov black flake8 isort mypy build wheel twine

echo -e "\n=== Development environment setup complete! ==="
echo "You can now run linting checks with: bash tools/lint.sh"
echo "Run tests with: pytest --cov=tsne_pso tests/" 
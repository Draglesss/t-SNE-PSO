#!/bin/bash
# Script to run linting checks locally
# Usage: bash tools/lint.sh

set -e  # Exit on error

# Ensure we're at the repo root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "=== Running Linting Checks ==="

# Function to check if a command exists
command_exists() {
  command -v "$1" >/dev/null 2>&1
}

# Check if dependencies are installed
for cmd in flake8 black isort mypy; do
  if ! command_exists "$cmd"; then
    echo "Error: $cmd is not installed. Install with:"
    echo "  pip install $cmd"
    exit 1
  fi
done

echo "=== Running flake8 ==="
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
# Exit-zero treats all errors as warnings
flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

echo "=== Running black ==="
black --check .

echo "=== Running isort ==="
isort --check --profile black .

echo "=== Running mypy ==="
mypy --ignore-missing-imports tsne_pso/

echo -e "\n=== All linting checks passed! ===" 
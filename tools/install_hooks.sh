#!/bin/bash
# Script to install git hooks
# Usage: bash tools/install_hooks.sh

set -e  # Exit on error

# Ensure we're at the repo root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "=== Installing git hooks ==="

# Create pre-commit hook
cat > .git/hooks/pre-commit << 'EOF'
#!/bin/bash

# Path to repo root
REPO_ROOT="$(git rev-parse --show-toplevel)"

# Run linting checks
bash "$REPO_ROOT/tools/lint.sh"

# If lint checks fail, prevent the commit
if [ $? -ne 0 ]; then
  echo "Linting failed! Fix the issues before committing."
  exit 1
fi

exit 0
EOF

# Make the pre-commit hook executable
chmod +x .git/hooks/pre-commit

echo -e "\n=== Git hooks installed successfully! ==="
echo "The pre-commit hook will run linting checks before each commit." 
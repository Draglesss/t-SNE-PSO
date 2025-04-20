# PowerShell script to install git hooks
# Usage: .\tools\install_hooks.ps1

# Ensure we're at the repo root
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location "$ScriptDir\.."

Write-Host "=== Installing git hooks ===" -ForegroundColor Cyan

# Get the repo root
$RepoRoot = git rev-parse --show-toplevel

# Create pre-commit hook
$PreCommitHook = @"
#!/bin/bash

# Path to repo root
REPO_ROOT="`$(git rev-parse --show-toplevel)"

# Check if running on Windows
if [ -x "`$(command -v powershell)" ]; then
  # Use PowerShell script on Windows
  powershell.exe -ExecutionPolicy Bypass -File "`$REPO_ROOT/tools/lint.ps1"
else
  # Use Bash script on Unix-like systems
  bash "`$REPO_ROOT/tools/lint.sh"
fi

# If lint checks fail, prevent the commit
if [ `$? -ne 0 ]; then
  echo "Linting failed! Fix the issues before committing."
  exit 1
fi

exit 0
"@

# Ensure the hooks directory exists
if (-not (Test-Path ".git/hooks")) {
    New-Item -ItemType Directory -Path ".git/hooks" -Force | Out-Null
}

# Write the pre-commit hook
Set-Content -Path ".git/hooks/pre-commit" -Value $PreCommitHook

# Make the pre-commit hook executable (doesn't matter on Windows but helps with WSL)
if (Test-Path "/bin/chmod") {
    & /bin/chmod +x .git/hooks/pre-commit
} else {
    Write-Host "Note: Unable to make the hook executable. If using WSL or Git Bash, run: chmod +x .git/hooks/pre-commit" -ForegroundColor Yellow
}

Write-Host "`n=== Git hooks installed successfully! ===" -ForegroundColor Green
Write-Host "The pre-commit hook will run linting checks before each commit." -ForegroundColor Yellow 
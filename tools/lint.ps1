# PowerShell script to run linting checks locally
# Usage: .\tools\lint.ps1

# Ensure we're at the repo root
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location "$ScriptDir\.."

Write-Host "=== Running Linting Checks ===" -ForegroundColor Cyan

# Function to check if a command exists
function Test-CommandExists {
    param ($command)
    try {
        if (Get-Command $command -ErrorAction Stop) {
            return $true
        }
    }
    catch {
        return $false
    }
}

# Check if dependencies are installed
$deps = @("flake8", "black", "isort", "mypy")
foreach ($cmd in $deps) {
    if (-not (Test-CommandExists $cmd)) {
        Write-Host "Error: $cmd is not installed. Install with:" -ForegroundColor Red
        Write-Host "  pip install $cmd" -ForegroundColor Yellow
        exit 1
    }
}

Write-Host "=== Running flake8 ===" -ForegroundColor Cyan
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
# Exit-zero treats all errors as warnings
flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

Write-Host "=== Running black ===" -ForegroundColor Cyan
black --check .

Write-Host "=== Running isort ===" -ForegroundColor Cyan
isort --check --profile black .

Write-Host "=== Running mypy ===" -ForegroundColor Cyan
mypy --ignore-missing-imports tsne_pso/

Write-Host "`n=== All linting checks passed! ===" -ForegroundColor Green 
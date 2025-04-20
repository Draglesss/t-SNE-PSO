# PowerShell script to set up development environment
# Usage: .\tools\setup_dev.ps1

# Ensure we're at the repo root
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location "$ScriptDir\.."

Write-Host "=== Setting up development environment ===" -ForegroundColor Cyan

# Install package in development mode
python -m pip install -e .

# Install development dependencies
python -m pip install --upgrade pip
python -m pip install pytest pytest-cov black flake8 isort mypy build wheel twine

Write-Host "`n=== Development environment setup complete! ===" -ForegroundColor Green
Write-Host "You can now run linting checks with: .\tools\lint.ps1" -ForegroundColor Yellow
Write-Host "Run tests with: pytest --cov=tsne_pso tests/" -ForegroundColor Yellow 
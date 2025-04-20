# Development Tools

This directory contains scripts to help with development tasks for t-SNE-PSO.

## Setup

### Linux/macOS

```bash
# Install development dependencies
bash tools/setup_dev.sh

# Install git hooks
bash tools/install_hooks.sh
```

### Windows

```powershell
# Install development dependencies
.\tools\setup_dev.ps1

# Install git hooks
.\tools\install_hooks.ps1
```

## Linting

These scripts run style checks to ensure code quality.

### Linux/macOS

```bash
bash tools/lint.sh
```

### Windows

```powershell
.\tools\lint.ps1
```

## Pre-commit Hooks

After installing the git hooks, linting checks will automatically run before each commit to ensure that your code meets the project's style guidelines. This helps maintain code quality and consistency.

## CI/CD Workflow

The GitHub Actions workflow in `.github/workflows/python-package.yml` automatically runs tests and linting on multiple platforms (Windows, Linux, macOS) and Python versions when you push to the repository. It:

1. Runs linting checks
2. Runs tests on multiple platforms and Python versions
3. Builds the package to ensure it can be distributed

## Manual Testing

To run tests manually:

```bash
# Run all tests with coverage
pytest --cov=tsne_pso tests/

# Run a specific test file
pytest tests/test_tsne_pso.py
``` 
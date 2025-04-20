# Contributing to TSNE-PSO

Thank you for your interest in contributing to TSNE-PSO! This document provides guidelines and instructions for contributing.

## Code of Conduct

Please be respectful and considerate of others when contributing to this project. Ensure all interactions are professional and constructive.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally
3. Set up the development environment
4. Create a branch for your work

```bash
# Clone your fork
git clone https://github.com/dragless/t-SNE-PSO.git
cd t-SNE-PSO

# Set up remote upstream
git remote add upstream https://github.com/dragless/t-SNE-PSO.git
```

## Setting up the Development Environment

### Linux/macOS

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate

# Set up using the provided script
bash tools/setup_dev.sh

# Install git hooks (optional but recommended)
bash tools/install_hooks.sh
```

### Windows

```powershell
# Create a virtual environment
python -m venv venv
.\venv\Scripts\activate

# Set up using the provided script
.\tools\setup_dev.ps1

# Install git hooks (optional but recommended)
.\tools\install_hooks.ps1
```

## Making Changes

1. Make your changes to the codebase
2. Add tests for any new features
3. Ensure all tests pass
4. Update documentation if necessary
5. Follow the code style guidelines

### Code Style

- Follow PEP 8 style guidelines
- Use descriptive variable names
- Keep functions small and focused
- Add appropriate docstrings following NumPy/SciPy conventions
- Use type annotations where appropriate

### Linting

Run linting checks to ensure your code meets style guidelines:

```bash
# Linux/macOS
bash tools/lint.sh

# Windows
.\tools\lint.ps1
```

If you've installed the git hooks, linting will automatically run before each commit.

## Testing

Run the tests to ensure your changes don't break existing functionality:

```bash
# Run with coverage
pytest --cov=tsne_pso tests/

# Run specific tests
pytest tests/test_tsne_pso.py
```

## Submitting Changes

1. Commit your changes with clear, descriptive commit messages
2. Push your branch to your fork
3. Submit a pull request to the main repository

```bash
git add .
git commit -m "Add feature: clear description of changes"
git push origin feature/your-feature-name
```

## Pull Request Process

1. Ensure your PR addresses a specific issue or has a clear purpose
2. Include a description of the changes and their purpose
3. Update documentation if necessary
4. Request review from a maintainer
5. Address any feedback or requested changes

## Reporting Issues

If you find bugs or have feature requests, please create an issue on GitHub:

1. Check if the issue already exists
2. Use a clear, descriptive title
3. Provide detailed steps to reproduce the issue
4. Include relevant information (environment, version, etc.)

## Feature Requests

Feature requests are welcome. Please provide:

1. A clear description of the feature
2. The motivation for the feature
3. Possible implementation approaches if you have them

## Continuous Integration

The project uses GitHub Actions for CI/CD. When you submit a pull request, automated tests will run on multiple operating systems (Windows, Linux, macOS) and Python versions. Make sure your code passes all checks before requesting a review.

Thank you for contributing to TSNE-PSO! 
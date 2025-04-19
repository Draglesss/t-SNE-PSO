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


## Setting up the Development Environment

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"
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

## Testing

Run the tests to ensure your changes don't break existing functionality:

```bash
pytest
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

Thank you for contributing to TSNE-PSO! 
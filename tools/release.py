#!/usr/bin/env python
"""Release script for tsne_pso package.

This script automates the release process for the tsne_pso package.
It updates version numbers, creates git tags, and pushes to PyPI.
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path


def update_version(new_version):
    """Update version in tsne_pso/_version.py."""
    version_file = Path("tsne_pso/_version.py")
    content = version_file.read_text()
    new_content = re.sub(
        r'__version__ = "[\d.]+[a-z0-9]*"',
        f'__version__ = "{new_version}"',
        content,
    )
    version_file.write_text(new_content)
    print(f"Updated version to {new_version} in {version_file}")


def update_tests(new_version):
    """Update version check in tests."""
    test_file = Path("tests/test_basic.py")
    content = test_file.read_text()
    new_content = re.sub(
        r'assert tsne_pso.__version__ == "[\d.]+[a-z0-9]*"',
        f'assert tsne_pso.__version__ == "{new_version}"',
        content,
    )
    test_file.write_text(new_content)
    print(f"Updated version check in {test_file}")


def run_tests():
    """Run tests to ensure everything is working correctly."""
    print("Running tests...")
    result = subprocess.run(["pytest", "-v"], capture_output=True, text=True)
    if result.returncode != 0:
        print("Tests failed:")
        print(result.stdout)
        print(result.stderr)
        sys.exit(1)
    print("Tests passed.")


def build_package():
    """Build the package."""
    print("Building package...")
    subprocess.run(["python", "-m", "build"], check=True)
    print("Package built successfully.")


def create_git_tag(version):
    """Create a git tag for the release."""
    print(f"Creating git tag v{version}...")
    subprocess.run(["git", "add", "."], check=True)
    subprocess.run(["git", "commit", "-m", f"Release v{version}"], check=True)
    subprocess.run(["git", "tag", f"v{version}"], check=True)
    print("Git tag created.")


def push_to_github(version):
    """Push changes and tags to GitHub."""
    print("Pushing to GitHub...")
    subprocess.run(["git", "push"], check=True)
    subprocess.run(["git", "push", "origin", f"v{version}"], check=True)
    print("Changes pushed to GitHub.")


def upload_to_pypi():
    """Upload the package to PyPI."""
    print("Uploading to PyPI...")
    subprocess.run(["python", "-m", "twine", "upload", "dist/*"], check=True)
    print("Package uploaded to PyPI.")


def main():
    """Main entry point for the release script."""
    parser = argparse.ArgumentParser(description="Release a new version of tsne_pso")
    parser.add_argument("version", help="The new version number (e.g., 1.2.0)")
    parser.add_argument("--skip-tests", action="store_true", help="Skip running tests")
    parser.add_argument("--skip-git", action="store_true", help="Skip git operations")
    parser.add_argument(
        "--skip-pypi", action="store_true", help="Skip uploading to PyPI"
    )

    args = parser.parse_args()

    # Validate version format
    if not re.match(r"^\d+\.\d+\.\d+([a-z0-9]+)?$", args.version):
        print("Invalid version format. Expected format: X.Y.Z[suffix]")
        sys.exit(1)

    # Update version in files
    update_version(args.version)
    update_tests(args.version)

    # Run tests
    if not args.skip_tests:
        run_tests()

    # Build package
    build_package()

    # Git operations
    if not args.skip_git:
        create_git_tag(args.version)
        push_to_github(args.version)

    # Upload to PyPI
    if not args.skip_pypi:
        upload_to_pypi()

    print(f"Release v{args.version} completed successfully!")


if __name__ == "__main__":
    main()

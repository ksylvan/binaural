"""Conftest.py for pytest configuration."""

import os
import sys

# Add the project root to sys.path so that the binaural package is discoverable.
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


# Add command line options
def pytest_addoption(parser):
    """Add custom command line options to pytest."""
    parser.addoption(
        "--run-performance",
        action="store_true",
        default=False,
        help="Run performance tests",
    )


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "performance: mark test as a performance test")

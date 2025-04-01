"""Conftest.py for pytest configuration."""

import os
import sys

# Add the project root to sys.path so that the binaural package is discoverable.
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

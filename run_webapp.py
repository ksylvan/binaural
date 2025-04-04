#!/usr/bin/env python
"""Run the Binaural Beat Generator Streamlit web application."""

import sys
from pathlib import Path

import streamlit.web.cli as stcli


def main():
    """Run the Streamlit app."""
    # Get the directory of this script
    script_dir = Path(__file__).parent.absolute()

    # Path to the Streamlit app
    app_path = script_dir / "binaural" / "webui.py"

    # Check if the file exists
    if not app_path.exists():
        print(f"Error: Could not find Streamlit app at {app_path}")
        sys.exit(1)

    # Construct arguments for Streamlit CLI
    sys.argv = [
        "streamlit",
        "run",
        str(app_path),
        "--server.port=8501",
        "--server.address=0.0.0.0",
    ]

    # Run the Streamlit CLI with the constructed arguments
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()

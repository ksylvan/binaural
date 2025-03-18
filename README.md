# Binaural

Binaural is a Python tool that generates binaural beat audio designed to induce different brain wave states.

## Description

This tool creates a WAV file that transitions through different brain wave frequency bands using binaural beats:

- Beta (18 Hz) - 3 minutes - Associated with alertness and concentration
- Alpha (transition from 18 Hz to 10 Hz) - 5 minutes - Associated with relaxation and calmness
- Theta (transition from 10 Hz to 6 Hz) - 5 minutes - Associated with deep meditation and light sleep
- Delta (transition from 2 Hz to 6 Hz) - 7 minutes - Associated with deep sleep and healing

The program uses a 100 Hz base carrier frequency and creates stereo audio where the frequency difference between left and right channels creates the binaural beat effect.

## Installation

### Requirements

- Python 3.x
- Dependencies listed in requirements.txt:
  - numpy
  - wave

### Setup

1. Automatic setup with the provided script:

    ```bash
    ./bin/setup.sh
    ```

    This script will:

   - Install uv (if not already installed)
   - Create a virtual environment
   - Install all required dependencies

2. Activate the virtual environment:

    ```bash
    source .venv/bin/activate
    ```

    > Note: If using VS Code, the workspace is configured to run the setup script automatically when opening the folder.

## Usage

Run the script to generate a binaural beat WAV file:

```bash
python binaural.py
```

This will create `binaural_beats.wav` in the current directory.

## File Structure

- binaural.py - Main script that generates the binaural beats audio
- bin/setup.sh - Setup script to prepare the development environment
- requirements.txt - Python dependencies
- requirements-bootstrap.txt - Bootstrap dependencies for setup

## License

This project is licensed under the MIT License - see the LICENSE file for details.

Copyright (c) 2025 Kayvan Sylvan

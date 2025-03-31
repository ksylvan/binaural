"""Command-line interface for generating binaural beats audio from a YAML script."""

import argparse
import sys
import logging

from binaural.constants import (
    DEFAULT_BASE_FREQUENCY,
    DEFAULT_OUTPUT_FILENAME,
    DEFAULT_SAMPLE_RATE,
)
from binaural.tone_generator import generate_audio_sequence, save_audio_file
from binaural.utils import load_yaml_config
from binaural.exceptions import BinauralError


def main() -> None:
    """Main function to parse command line arguments and generate binaural beats audio."""
    parser = argparse.ArgumentParser(
        description="Generate binaural beats audio (WAV or FLAC) from a YAML script."
    )
    parser.add_argument("script", help="Path to YAML configuration script.")
    parser.add_argument(
        "-o", "--output", help="Output audio file path (overrides YAML setting)."
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose logging."
    )
    args = parser.parse_args()

    # Set up logging configuration based on verbose flag
    logging_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=logging_level, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)

    try:
        config = load_yaml_config(args.script)
        sample_rate = config.get("sample_rate", DEFAULT_SAMPLE_RATE)
        base_freq = config.get("base_frequency", DEFAULT_BASE_FREQUENCY)
        output_filename = args.output or config.get(
            "output_filename", DEFAULT_OUTPUT_FILENAME
        )
        logger.debug("Configuration loaded: %s", config)

        left_channel, right_channel, total_duration = generate_audio_sequence(
            sample_rate, base_freq, config["steps"]
        )
        logger.info("Audio sequence generated successfully.")

        save_audio_file(
            output_filename, sample_rate, left_channel, right_channel, total_duration
        )
        logger.info("Audio file saved successfully.")
    except BinauralError as e:
        logger.error("Error: %s", e)
        sys.exit(f"Error: {e}")


if __name__ == "__main__":
    main()

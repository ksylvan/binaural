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
from binaural.data_types import NoiseConfig


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate binaural beats audio (WAV or FLAC) from a YAML script."
    )
    parser.add_argument("script", help="Path to YAML configuration script.")
    parser.add_argument(
        "-o", "--output", help="Output audio file path (overrides YAML setting)."
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging."
    )
    return parser.parse_args()


def configure_logging(verbose: bool) -> None:
    """Configure logging based on verbosity."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def main() -> None:
    """Main entry point for the CLI."""
    args = parse_args()
    configure_logging(args.verbose)
    logger = logging.getLogger(__name__)

    try:
        # Load YAML configuration using the utility function
        config = load_yaml_config(args.script)

        # Extract global settings, providing defaults if they are missing
        sample_rate = config.get("sample_rate", DEFAULT_SAMPLE_RATE)
        base_freq = config.get("base_frequency", DEFAULT_BASE_FREQUENCY)

        # Output file: use command-line override first, then YAML, then default
        output_filename = args.output or config.get(
            "output_filename", DEFAULT_OUTPUT_FILENAME
        )

        # Extract the noise configuration (guaranteed to exist by load_yaml_config)
        noise_config = config.get("noise_config", NoiseConfig())  # Default just in case

        logger.debug("Loaded configuration: %s", config)
        logger.info("Sample Rate: %d Hz", sample_rate)
        logger.info("Base Frequency: %.2f Hz", base_freq)
        if noise_config.type != "none":
            logger.info(
                "Background Noise: Type='%s', Amplitude=%.2f",
                noise_config.type,
                noise_config.amplitude,
            )

        # Generate the audio sequence using the tone generator module
        left_channel, right_channel, total_duration = generate_audio_sequence(
            sample_rate=sample_rate,
            base_freq=base_freq,
            steps=config["steps"],
            noise_config=noise_config,
        )

        logger.info("Audio sequence generated successfully.")

        # Save the generated audio to the specified file
        save_audio_file(
            filename=output_filename,
            sample_rate=sample_rate,
            left=left_channel,
            right=right_channel,
            total_duration_sec=total_duration,
        )

        logger.info("Audio file saved successfully to '%s'.", output_filename)

    except BinauralError as e:
        # Handle specific Binaural errors gracefully
        logger.error("Error: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()

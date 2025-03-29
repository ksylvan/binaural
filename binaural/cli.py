"""Command-line interface for generating binaural beats audio from a YAML script."""

import argparse
from binaural.utils import load_yaml_config
from binaural.tone_generator import generate_audio_sequence, save_audio_file
from binaural.constants import (
    DEFAULT_SAMPLE_RATE,
    DEFAULT_BASE_FREQUENCY,
    DEFAULT_OUTPUT_FILENAME,
)


def main() -> None:
    """Main function to parse command line arguments and generate binaural beats audio."""
    parser = argparse.ArgumentParser(
        description="Generate binaural beats audio (WAV or FLAC) from a YAML script."
    )
    parser.add_argument("script", help="Path to YAML configuration script.")
    parser.add_argument(
        "-o",
        "--output",
        help="Output audio file path (overrides YAML setting).",
    )
    args = parser.parse_args()

    config = load_yaml_config(args.script)

    sample_rate = config.get("sample_rate", DEFAULT_SAMPLE_RATE)
    base_freq = config.get("base_frequency", DEFAULT_BASE_FREQUENCY)
    output_filename = args.output or config.get(
        "output_filename", DEFAULT_OUTPUT_FILENAME
    )

    left_channel, right_channel, total_duration = generate_audio_sequence(
        sample_rate, base_freq, config["steps"]
    )

    save_audio_file(
        output_filename, sample_rate, left_channel, right_channel, total_duration
    )


if __name__ == "__main__":
    main()

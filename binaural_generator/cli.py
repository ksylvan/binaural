"""Command-line interface for generating binaural beats audio from a YAML script."""

import argparse
import logging
import multiprocessing
import os
from pathlib import Path
import sys
import time
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from binaural_generator.core.constants import (
    DEFAULT_BASE_FREQUENCY,
    DEFAULT_OUTPUT_FILENAME,
    DEFAULT_SAMPLE_RATE,
)
from binaural_generator.core.data_types import NoiseConfig
from binaural_generator.core.exceptions import BinauralError
from binaural_generator.core.parallel import generate_audio_sequence_parallel
from binaural_generator.core.tone_generator import (
    generate_audio_sequence,
    save_audio_file,
)
from binaural_generator.core.utils import get_all_script_configs, load_yaml_config
from binaural_generator import __version__


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate binaural beats audio from a YAML script."
    )
    cmd_group = parser.add_argument_group(
        title="Audio Generation Options",
        description="Options for generating audio from a YAML script.",
    )
    cmd_group.add_argument(
        "-o", "--output", help="Output audio file path (overrides YAML setting)."
    )
    cmd_group.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging."
    )
    cmd_group.add_argument(
        "-p",
        "--parallel",
        action="store_true",
        help="Use parallel processing for faster audio generation.",
    )
    cmd_group.add_argument(
        "--threads",
        type=int,
        help="Number of threads to use for parallel processing. Defaults to CPU count.",
        default=multiprocessing.cpu_count(),
    )
    cmd_group.add_argument(
        "script", help="Path to YAML configuration script.", nargs="?"
    )

    info_group = parser.add_argument_group(
        title="Information Options",
        description="Options for displaying information about the script.",
    )

    list_parser_group = info_group.add_mutually_exclusive_group()
    list_parser_group.add_argument(
        "--version", action="version", version=f"binaural-generator {__version__}"
    )
    list_parser_group.add_argument(
        "-l", "--list", action="store_true", help="List available scripts."
    )
    return parser.parse_args()


def configure_logging(verbose: bool) -> None:
    """Configure logging based on verbosity."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


@dataclass
class AudioGenerationResult:
    """Class to hold the result of audio generation.

    Attributes:
        left_channel: Numpy array for the left audio channel.
        right_channel: Numpy array for the right audio channel.
        total_duration: The total duration of the audio in seconds.
        processing_time: The time taken to generate the audio in seconds.
    """

    left_channel: np.ndarray
    right_channel: np.ndarray
    total_duration: float
    processing_time: float

    def __str__(self) -> str:
        """Return a string representation of the results."""
        minutes, seconds = divmod(self.total_duration, 60)
        return (
            f"Audio generation completed in {self.processing_time:.2f}s. "
            f"Total duration: {int(minutes)}m {seconds:.2f}s. "
            f"Left channel: {len(self.left_channel)} samples, "
            f"Right channel: {len(self.right_channel)} samples."
        )


@dataclass
class AudioConfig:
    """Configuration for audio generation.

    Attributes:
        sample_rate: The audio sample rate in Hz.
        base_freq: The base carrier frequency in Hz.
        steps: A list of dictionaries, each representing an audio generation step.
        noise_config: A NoiseConfig object specifying background noise settings.
        title: The title of the audio session.
        use_parallel: Whether to use parallel processing.
        max_workers: Maximum number of worker threads (None = use CPU count).
    """

    sample_rate: int
    base_freq: float
    steps: list[dict[str, Any]]
    noise_config: NoiseConfig
    title: str = "Binaural Beat"
    use_parallel: bool = False
    max_workers: Optional[int] = None


def generate_audio(config: AudioConfig) -> AudioGenerationResult:
    """Generate audio using either sequential or parallel processing.

    Args:
        config: AudioConfig containing all generation parameters

    Returns:
        An AudioGenerationResult object containing the generated audio data
        and processing information.
    """
    logger = logging.getLogger(__name__)
    start_time = time.time()

    if config.use_parallel:
        logger.info("Using parallel processing for audio generation...")
        left_channel, right_channel, total_duration = generate_audio_sequence_parallel(
            sample_rate=config.sample_rate,
            base_freq=config.base_freq,
            steps=config.steps,
            noise_config=config.noise_config,
            title=config.title,
            max_workers=config.max_workers,
        )
    else:
        logger.info("Using sequential processing for audio generation...")
        left_channel, right_channel, total_duration = generate_audio_sequence(
            sample_rate=config.sample_rate,
            base_freq=config.base_freq,
            steps=config.steps,
            noise_config=config.noise_config,
            title=config.title,
        )

    processing_time = time.time() - start_time

    return AudioGenerationResult(
        left_channel=left_channel,
        right_channel=right_channel,
        total_duration=total_duration,
        processing_time=processing_time,
    )


def main() -> None:
    """Main entry point for the CLI."""
    args = parse_args()
    configure_logging(args.verbose)
    logger = logging.getLogger(__name__)

    script_dir = Path(__file__).parent.absolute()

    # Determine script path
    script_path = args.script
    if script_path and not os.path.exists(args.script):
        # If not found and no path separators in the name, check in scripts directory
        if os.path.basename(args.script) == args.script:
            potential_script_path = script_dir / "scripts" / args.script
            if os.path.exists(potential_script_path):
                script_path = potential_script_path
                logger.info("Using script from scripts directory: %s", script_path)

    if args.list:
        os.chdir(script_dir)
        all_scripts = get_all_script_configs(file_names_only=True)
        print(
            f"Available scripts: in {script_dir / 'scripts'}\n\n"
            + "\n".join(
                sorted([f"  {title}: {path}" for title, path in all_scripts.items()])
            )
        )
        print("\nUse --script <script_name.yaml> to run a specific script.")
        sys.exit(0)

    if not script_path:
        logger.error("No script specified. Use --script <script_name.yaml>.")
        sys.exit(1)

    try:
        # Load YAML configuration using the utility function
        config = load_yaml_config(script_path)

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
        logger.info("Processing Audio for: %s", config.get("title", "Binaural Beat"))
        logger.info("Sample Rate: %d Hz", sample_rate)
        logger.info("Base Frequency: %.2f Hz", base_freq)
        if noise_config.type != "none":
            logger.info(
                "Background Noise: Type='%s', Amplitude=%.2f",
                noise_config.type,
                noise_config.amplitude,
            )

        # Create audio configuration
        audio_config = AudioConfig(
            sample_rate=sample_rate,
            base_freq=base_freq,
            steps=config["steps"],
            noise_config=noise_config,
            title=config.get("title", "Binaural Beat"),  # Get title from config
            use_parallel=args.parallel,
            max_workers=args.threads,
        )

        # Generate the audio sequence using either parallel or sequential processing
        result = generate_audio(audio_config)

        logger.info(
            "Audio sequence generated successfully in %.2f seconds.",
            result.processing_time,
        )

        # Save the generated audio to the specified file
        save_audio_file(
            filename=output_filename,
            sample_rate=sample_rate,
            left=result.left_channel,
            right=result.right_channel,
            total_duration_sec=result.total_duration,
        )

        logger.info("Audio file saved successfully to '%s'.", output_filename)

    except BinauralError as e:
        # Handle specific Binaural errors gracefully
        logger.error("Error: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()

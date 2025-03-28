#!/usr/bin/env python

"""Binaural Beat Generator

Generates binaural beat audio (WAV or FLAC) from a YAML configuration file.
"""

import argparse
import os
import sys
from typing import List, Tuple

import numpy as np
import soundfile as sf
import yaml

DEFAULT_SAMPLE_RATE = 44100
DEFAULT_BASE_FREQUENCY = 100
DEFAULT_OUTPUT_FILENAME = "output.flac"  # Default format is FLAC
SUPPORTED_FORMATS = (".wav", ".flac")


def generate_tone(
    duration_sec: float,
    base_freq: float,
    freq_diff_start: float,
    freq_diff_end: float,
    sample_rate: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generates stereo audio data for binaural beats."""
    # Calculate the number of samples required for the duration
    num_samples = int(sample_rate * duration_sec)
    # Create a time vector from 0 to duration_sec with num_samples points
    t = np.linspace(0, duration_sec, num_samples, endpoint=False)
    # Create a frequency difference vector linearly interpolating from start to end
    freq_diff = np.linspace(freq_diff_start, freq_diff_end, num_samples)

    # Generate the left channel sine wave at the base frequency
    left_channel = np.sin(2 * np.pi * base_freq * t)
    # Generate the right channel sine wave at the base frequency plus the difference
    right_channel = np.sin(2 * np.pi * (base_freq + freq_diff) * t)

    # Return the generated left and right channels
    return left_channel, right_channel


def load_yaml_config(path: str) -> dict:
    """Loads and validates YAML configuration."""
    try:
        # Open and read the YAML file
        with open(path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
        # Basic validation: ensure it's a dictionary with a 'steps' key
        if not isinstance(config, dict) or "steps" not in config:
            raise ValueError("YAML must have a 'steps' key with a list of steps.")
        # Return the loaded configuration
        return config
    except FileNotFoundError:
        # Handle file not found error
        sys.exit(f"Error: Config file '{path}' not found.")
    except yaml.YAMLError as e:
        # Handle YAML parsing errors
        sys.exit(f"Error parsing YAML file '{path}': {e}")
    except ValueError as e:
        # Handle configuration validation errors
        sys.exit(f"Configuration error: {e}")


def validate_step(step: dict, previous_freq: float) -> Tuple[str, float, float, float]:
    """Validates and extracts necessary fields from a step."""
    # Get step type and duration
    step_type = step.get("type")
    duration_min = step.get("duration")

    # Validate step type
    if step_type not in ("stable", "transition"):
        raise ValueError(
            f"Invalid step type '{step_type}'. Must be 'stable' or 'transition'."
        )
    # Validate duration
    if not isinstance(duration_min, (int, float)) or duration_min <= 0:
        raise ValueError("Step duration must be a positive number in minutes.")

    # Convert duration from minutes to seconds
    duration_sec = duration_min * 60

    # Handle 'stable' type steps
    if step_type == "stable":
        freq = step.get("frequency")
        # Validate frequency for stable step
        if not isinstance(freq, (int, float)):
            raise ValueError("Stable step must specify a valid 'frequency'.")
        # Return type, duration, and start/end frequency (same for stable)
        return step_type, duration_sec, freq, freq

    # Handle 'transition' type steps
    # Use previous step's end frequency if start_frequency is not provided
    start_freq = step.get("start_frequency", previous_freq)
    end_freq = step.get("end_frequency")

    # Validate start and end frequencies for transition step
    if start_freq is None:
        raise ValueError(
            "Transition step must specify 'start_frequency' if it's the first step or "
            "if the previous step's frequency is unknown."
        )
    if not all(isinstance(f, (int, float)) for f in (start_freq, end_freq)):
        raise ValueError(
            "Transition step must specify valid numeric 'start_frequency' "
            "(or have a previous step) and 'end_frequency'."
        )

    # Return type, duration, start frequency, and end frequency
    return step_type, duration_sec, start_freq, end_freq


def generate_audio_sequence(
    steps: List[dict], base_freq: float, sample_rate: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Generates the complete audio sequence from the list of steps."""
    # Initialize lists to hold audio data for each channel
    left_audio, right_audio = [], []
    # Keep track of the end frequency of the last processed step
    previous_freq = None

    # Iterate through each step in the configuration
    for idx, step in enumerate(steps, start=1):
        try:
            # Validate the current step and get its parameters
            step_type, duration_sec, freq_start, freq_end = validate_step(
                step, previous_freq
            )
            # Print progress information
            print(
                f"Generating step {idx}: {step_type}, "
                f"{freq_start}Hz -> {freq_end}Hz, duration {duration_sec / 60:.2f}min"
            )
            # Generate the audio tones for the current step
            left, right = generate_tone(
                duration_sec, base_freq, freq_start, freq_end, sample_rate
            )
            # Append the generated audio data to the respective channel lists
            left_audio.append(left)
            right_audio.append(right)
            # Update the previous frequency for the next step's potential transition
            previous_freq = freq_end
        except ValueError as e:
            # Handle validation errors for the specific step
            sys.exit(f"Error in step {idx}: {e}")

    # Concatenate all audio segments into single arrays for each channel
    return np.concatenate(left_audio), np.concatenate(right_audio)


def save_audio_file(
    filename: str, left: np.ndarray, right: np.ndarray, sample_rate: int
):
    """Saves stereo audio data to a WAV or FLAC file using soundfile."""
    # Check if the filename has a supported extension
    _root, ext = os.path.splitext(filename)
    if ext.lower() not in SUPPORTED_FORMATS:
        sys.exit(
            f"Error: Unsupported output file format '{ext}'. "
            f"Supported formats are: {', '.join(SUPPORTED_FORMATS)}"
        )

    # Stack the left and right channels vertically and transpose for
    # stereo format (samples, channels)
    stereo_audio = np.vstack((left, right)).T

    # Ensure the output directory exists
    output_dir = os.path.dirname(filename)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
        except OSError as e:
            sys.exit(f"Error creating output directory '{output_dir}': {e}")

    try:
        # Write the stereo audio data to the file using soundfile
        # soundfile handles normalization and format detection based on extension
        # subtype='PCM_16' is common for WAV, FLAC uses lossless compression by default
        sf.write(filename, stereo_audio, sample_rate, subtype="PCM_16")
        print(f"Audio file f'{filename}' created successfully.")
    except (sf.SoundFileError, RuntimeError, IOError) as e:
        # Handle errors during file writing
        sys.exit(f"Error writing audio file '{filename}': {e}")


def main(script_path: str, output_path: str = None):
    """Main function to generate binaural beats from a YAML script."""
    # Load the configuration from the YAML script
    config = load_yaml_config(script_path)

    # Get global settings from config or use defaults
    base_freq = config.get("base_frequency", DEFAULT_BASE_FREQUENCY)
    sample_rate = config.get("sample_rate", DEFAULT_SAMPLE_RATE)

    # Determine the output filename:
    # 1. Use the command-line override if provided.
    # 2. Otherwise, use the filename from the YAML config.
    # 3. Fallback to the default filename if neither is specified.
    output_filename = output_path or config.get(
        "output_filename", DEFAULT_OUTPUT_FILENAME
    )

    # Generate the complete audio sequence based on the steps
    left_channel, right_channel = generate_audio_sequence(
        config["steps"], base_freq, sample_rate
    )

    # Save the generated audio to the specified file
    save_audio_file(output_filename, left_channel, right_channel, sample_rate)


if __name__ == "__main__":
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(
        description="Generate binaural beats audio (WAV or FLAC) from a YAML script."
    )
    # Required argument: path to the YAML script
    parser.add_argument("script", help="Path to the YAML configuration script.")
    # Optional argument: override the output file path
    parser.add_argument(
        "-o",
        "--output",
        help="Output audio file path (e.g., 'output.wav' or 'output.flac'). "
        "Overrides 'output_filename' in the YAML script. "
        f"The file extension determines the format. Default is {DEFAULT_OUTPUT_FILENAME}.",
        default=DEFAULT_OUTPUT_FILENAME,
    )
    # Parse the command-line arguments
    args = parser.parse_args()

    # Run the main function with the parsed arguments
    main(args.script, args.output)

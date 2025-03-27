#!/usr/bin/env python

"""Binaural Beat Generator

This module generates a binaural beat WAV file based on a sequence defined
in a YAML configuration file.

It reads a script specifying steps, which can be either holding a stable
binaural beat frequency or transitioning between frequencies over a duration.
"""

import argparse
import wave
import sys
from typing import List, Dict, Any, Tuple

import numpy as np
import yaml

# Default parameters (can be overridden in YAML)
DEFAULT_SAMPLE_RATE = 44100
DEFAULT_BASE_FREQ = 100
DEFAULT_OUTPUT_FILENAME = "binaural_beats.wav"


def generate_tone(
    duration_sec: float,
    base_freq: float,
    start_diff: float,
    end_diff: float,
    sample_rate: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate stereo tones for the given duration.

    Args:
        duration_sec: Duration of the segment in seconds.
        base_freq: Base carrier frequency (Hz).
        start_diff: Starting frequency difference for the binaural beat (Hz).
        end_diff: Ending frequency difference for the binaural beat (Hz).
        sample_rate: Audio sample rate (samples per second).

    Returns:
        A tuple containing the left and right channel audio data as numpy arrays.
    """
    # Calculate the number of samples
    num_samples = int(sample_rate * duration_sec)
    # Create a time vector from 0 to duration_sec
    t = np.linspace(0, duration_sec, num_samples, endpoint=False)

    # Create a linear ramp for the binaural beat frequency difference
    diff = np.linspace(start_diff, end_diff, num_samples)

    # Generate the left channel signal (base frequency)
    left = np.sin(2 * np.pi * base_freq * t)
    # Generate the right channel signal (base frequency + difference)
    right = np.sin(2 * np.pi * (base_freq + diff) * t)

    return left, right


def load_script(script_path: str) -> Dict[str, Any]:
    """Load and parse the YAML script file.

    Args:
        script_path: Path to the YAML script file.

    Returns:
        A dictionary containing the parsed script configuration.

    Raises:
        FileNotFoundError: If the script file does not exist.
        yaml.YAMLError: If the script file is not valid YAML.
        ValueError: If the script format is invalid.
    """
    try:
        # Open and read the script file
        with open(script_path, "r", encoding="utf-8") as f:
            # Parse the YAML content safely
            script_data = yaml.safe_load(f)
        # Basic validation
        if not isinstance(script_data, dict) or "steps" not in script_data:
            raise ValueError(
                "Invalid script format: Must be a dictionary with a 'steps' key."
            )
        if not isinstance(script_data["steps"], list):
            raise ValueError("Invalid script format: 'steps' must be a list.")
        return script_data
    except FileNotFoundError:
        print(f"Error: Script file not found at '{script_path}'", file=sys.stderr)
        raise
    except yaml.YAMLError as e:
        print(f"Error parsing YAML script '{script_path}': {e}", file=sys.stderr)
        raise
    except ValueError as e:
        print(f"Error in script format '{script_path}': {e}", file=sys.stderr)
        raise


def main(script_path: str, output_filename_override: str = None):
    """Main function to load script, generate audio, and save WAV file.

    Args:
        script_path: Path to the YAML script file.
        output_filename_override: Optional path to override the output filename from the script.
    """
    try:
        # Load the script configuration from the YAML file
        script = load_script(script_path)
    except (FileNotFoundError, yaml.YAMLError, ValueError):
        # Exit if script loading fails
        sys.exit(1)

    # Get global settings from script or use defaults
    base_freq = script.get("base_frequency", DEFAULT_BASE_FREQ)
    sample_rate = script.get("sample_rate", DEFAULT_SAMPLE_RATE)
    output_filename = output_filename_override or script.get(
        "output_filename", DEFAULT_OUTPUT_FILENAME
    )

    # Initialize lists to hold audio data for each channel
    all_left_channel: List[np.ndarray] = []
    all_right_channel: List[np.ndarray] = []

    # Process each step defined in the script
    current_freq = None  # Keep track of the frequency at the end of the last step
    for i, step in enumerate(script.get("steps", [])):
        # Validate step type
        step_type = step.get("type")
        if step_type not in ["stable", "transition"]:
            print(
                f"Error in step {i+1}: Invalid type '{step_type}'."
                " Must be 'stable' or 'transition'.",
                file=sys.stderr,
            )
            sys.exit(1)

        # Get duration and convert from minutes to seconds
        duration_min = step.get("duration")
        if not isinstance(duration_min, (int, float)) or duration_min <= 0:
            print(
                f"Error in step {i+1}: Invalid or missing 'duration'."
                " Must be a positive number (minutes).",
                file=sys.stderr,
            )
            sys.exit(1)
        duration_sec = duration_min * 60

        # Initialize start and end frequencies for the tone generation
        start_diff = 0.0
        end_diff = 0.0

        # Process 'stable' step type
        if step_type == "stable":
            frequency = step.get("frequency")
            if not isinstance(frequency, (int, float)):
                print(
                    f"Error in step {i+1}: Invalid or missing 'frequency' for stable step.",
                    file=sys.stderr,
                )
                sys.exit(1)
            start_diff = frequency
            end_diff = frequency
            current_freq = frequency  # Update current frequency

        # Process 'transition' step type
        elif step_type == "transition":
            start_frequency = step.get("start_frequency")
            end_frequency = step.get("end_frequency")

            # Use current_freq if start_frequency is not provided for seamless transition
            if start_frequency is None:
                if current_freq is None:
                    print(
                        f"Error in step {i+1}: 'start_frequency' missing "
                        "and no previous step to transition from.",
                        file=sys.stderr,
                    )
                    sys.exit(1)
                start_frequency = current_freq

            if not isinstance(start_frequency, (int, float)) or not isinstance(
                end_frequency, (int, float)
            ):
                print(
                    f"Error in step {i+1}: Invalid or missing 'start_frequency' "
                    "or 'end_frequency' for transition step.",
                    file=sys.stderr,
                )
                sys.exit(1)
            start_diff = start_frequency
            end_diff = end_frequency
            current_freq = end_frequency  # Update current frequency

        # Generate the audio segment for this step
        print(
            f"Generating step {i+1}: type='{step_type}', "
            f"duration={duration_min} min, freq={start_diff:.1f}Hz -> {end_diff:.1f}Hz"
        )
        left_segment, right_segment = generate_tone(
            duration_sec, base_freq, start_diff, end_diff, sample_rate
        )

        # Append the generated segments to the lists
        all_left_channel.append(left_segment)
        all_right_channel.append(right_segment)

    # Check if any steps were processed
    if not all_left_channel:
        print(
            "Warning: No steps found in the script. Generating empty audio file.",
            file=sys.stderr,
        )
        left_channel = np.array([])
        right_channel = np.array([])
    else:
        # Concatenate all segments for each channel
        left_channel = np.concatenate(all_left_channel)
        right_channel = np.concatenate(all_right_channel)

    # Combine left and right channels into a stereo signal (interleaved)
    # Ensure the array is 2D with shape (N, 2)
    if left_channel.size > 0:
        stereo_signal = np.vstack((left_channel, right_channel)).T
    else:
        stereo_signal = np.empty((0, 2))

    # Normalize the audio signal to 16-bit integer range (-32767 to 32767)
    if stereo_signal.size > 0:
        max_val = np.max(np.abs(stereo_signal))
        if max_val > 0:
            # Avoid division by zero if signal is silence
            stereo_int16 = np.int16((stereo_signal / max_val) * 32767)
        else:
            stereo_int16 = np.int16(stereo_signal)  # Already zeros
    else:
        stereo_int16 = np.empty((0, 2), dtype=np.int16)

    # Write the combined audio data to a stereo WAV file
    print(f"Writing audio to '{output_filename}'...")
    try:
        with wave.open(output_filename, "wb") as wf:
            # TODO: #1 Remove the pylint disable once the issue is resolved
            # See https://github.com/microsoft/vscode-pylint/issues/603
            wf.setnchannels(2)  # Stereo # pylint: disable=no-member
            wf.setsampwidth(  # pylint: disable=no-member
                2
            )  # 16 bits per sample (2 bytes)
            wf.setframerate(sample_rate)  # pylint: disable=no-member
            # Write the frames as bytes
            wf.writeframes(stereo_int16.tobytes())  # pylint: disable=no-member
        print(f"Audio file '{output_filename}' generated successfully.")
    except (wave.Error, OSError, IOError) as e:
        print(f"Error writing WAV file '{output_filename}': {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Generate binaural beats audio from a YAML script."
    )
    # Add argument for the script file path (required)
    parser.add_argument(
        "script",
        help="Path to the YAML script file defining the binaural beat sequence.",
    )
    # Add optional argument for the output file path
    parser.add_argument(
        "-o",
        "--output",
        help="Path to the output WAV file. Overrides 'output_filename' in the script.",
        default=None,
    )

    # Parse command-line arguments
    args = parser.parse_args()

    # Run the main function with parsed arguments
    main(args.script, args.output)

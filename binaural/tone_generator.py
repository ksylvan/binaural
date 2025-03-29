"""Generates stereo audio data for binaural beats with volume envelope."""

import math
import os
import sys
from typing import Tuple

import numpy as np
import soundfile as sf

from binaural.fade import apply_fade
from binaural.utils import validate_step
from binaural.constants import SUPPORTED_FORMATS


def generate_tone(
    duration_sec: float,
    base_freq: float,
    freq_diff_start: float,
    freq_diff_end: float,
    sample_rate: int,
    fade_in_sec: float = 0.0,
    fade_out_sec: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generates stereo audio data for binaural beats with volume envelope."""
    num_samples = int(sample_rate * duration_sec)
    if num_samples == 0:
        return np.array([]), np.array([])

    # Create a time vector from 0 to duration_sec with num_samples points
    t = np.linspace(0, duration_sec, num_samples, endpoint=False)
    # Create a frequency difference vector linearly interpolating from start to end
    freq_diff = np.linspace(freq_diff_start, freq_diff_end, num_samples)

    # Generate the left channel sine wave at the base frequency
    left_channel = np.sin(2 * np.pi * base_freq * t)
    # Generate the right channel sine wave at the base frequency plus the difference
    right_channel = np.sin(2 * np.pi * (base_freq + freq_diff) * t)

    left_channel = apply_fade(left_channel, sample_rate, fade_in_sec, fade_out_sec)
    right_channel = apply_fade(right_channel, sample_rate, fade_in_sec, fade_out_sec)

    return left_channel, right_channel


def generate_audio_sequence(
    steps: list[dict], base_freq: float, sample_rate: int
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Generates the complete audio sequence and returns total duration."""
    # Initialize lists to hold audio data for each channel
    left_audio, right_audio = [], []
    # Keep track of the end frequency of the last processed step
    previous_freq: float | None = None
    # Initialize total duration in seconds
    total_duration_sec = 0.0

    # Iterate through each step in the configuration
    for idx, step in enumerate(steps, start=1):
        try:
            # Validate the current step and get its parameters, including fades
            (
                step_type,
                duration_sec,
                freq_start,
                freq_end,
                fade_in_sec,
                fade_out_sec,
            ) = validate_step(step, previous_freq)

            # Add step duration to total duration
            total_duration_sec += duration_sec

            # Print progress information, including fade details if present
            fade_info = ""
            if fade_in_sec > 0:
                fade_info += f", fade-in {fade_in_sec/60.0:.2f}min"
            if fade_out_sec > 0:
                fade_info += f", fade-out {fade_out_sec/60.0:.2f}min"

            print(
                f"Generating step {idx}: {step_type}, "
                f"{freq_start}Hz -> {freq_end}Hz, duration {duration_sec / 60.0:.2f}min"
                f"{fade_info}"
            )

            # Generate the audio tones for the current step, applying fades
            left, right = generate_tone(
                duration_sec,
                base_freq,
                freq_start,
                freq_end,
                sample_rate,
                fade_in_sec,
                fade_out_sec,
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
    # Return concatenated audio and the total duration in seconds
    if not left_audio:  # Handle case with no valid steps
        return np.array([]), np.array([]), 0.0

    return (
        np.concatenate(left_audio),
        np.concatenate(right_audio),
        total_duration_sec,
    )


def save_audio_file(
    filename: str,
    left: np.ndarray,
    right: np.ndarray,
    sample_rate: int,
    total_duration_sec: float,
) -> None:
    """Saves stereo audio data and prints total duration."""
    # Check if the filename has a supported extension
    _root, ext = os.path.splitext(filename)
    if ext.lower() not in SUPPORTED_FORMATS:
        sys.exit(
            f"Error: Unsupported output file format '{ext}'. "
            f"Supported formats are: {', '.join(SUPPORTED_FORMATS)}"
        )

    # Check if there's any audio data to save
    if left.size == 0 or right.size == 0:
        print(
            "Warning: No audio data generated. Output file will be empty or not created."
        )
        # Depending on soundfile behavior, it might create an empty file or error.
        # We'll attempt to write, but handle potential errors.
        # If you prefer not to create an empty file, you could exit here.

    # Stack the left and right channels vertically and transpose for
    # stereo format (samples, channels)
    # Ensure they have the same length (should be guaranteed by generation)
    min_len = min(left.size, right.size)
    stereo_audio = np.vstack((left[:min_len], right[:min_len])).T

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

        # Calculate total minutes and remaining seconds
        total_minutes = math.floor(total_duration_sec / 60)
        remaining_seconds = total_duration_sec % 60

        # Print success message including the total duration
        print(
            f"Audio file '{filename}' created successfully. "
            f"Total duration: {total_minutes} minutes and {remaining_seconds:.2f} seconds."
        )
    except (sf.SoundFileError, RuntimeError, IOError) as e:
        # Handle errors during file writing
        sys.exit(f"Error writing audio file '{filename}': {e}")

"""Generates stereo audio data for binaural beats with volume envelope."""

import math
import os
import sys
from typing import Tuple

import numpy as np
import soundfile as sf

from binaural.constants import SUPPORTED_FORMATS
from binaural.fade import apply_fade
from binaural.types import AudioStep, Tone


def generate_tone(
    sample_rate: int, duration_sec: float, tone: Tone
) -> Tuple[np.ndarray, np.ndarray]:
    """Generates stereo audio data for binaural beats with volume envelope."""

    # Unpack the tone parameters
    base_freq = tone.base_freq
    freq_diff_start = tone.freq_diff_start
    freq_diff_end = tone.freq_diff_end
    fade_in_sec = tone.fade_in_sec
    fade_out_sec = tone.fade_out_sec

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
    sample_rate: int,
    base_freq: float,
    steps: list[dict],
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
            audio_step = AudioStep(**step)  # Unpack and validate the step
            if idx >= 1:
                if audio_step.type == "transition":
                    if previous_freq is None:
                        raise ValueError(
                            "Transition step must specify 'start_frequency' if it's the first step "
                            "or if the previous step's frequency is unknown."
                        )
                    if audio_step.start_frequency is None:
                        # Use the previous frequency if not specified
                        audio_step.start_frequency = previous_freq
                elif audio_step.type == "stable":
                    # For stable steps, set the previous frequency to the current frequency
                    previous_freq = audio_step.frequency

            # Add step duration to total duration
            total_duration_sec += audio_step.duration

            # Print progress information
            print(f"Generating step {idx}: {audio_step}")

            # Generate the audio tones for the current step, applying fades
            left, right = generate_tone(
                sample_rate,
                audio_step.duration,
                Tone(
                    base_freq,
                    audio_step.start_frequency,
                    audio_step.end_frequency,
                    audio_step.fade_in_duration,
                    audio_step.fade_out_duration,
                ),
            )
            # Append the generated audio data to the respective channel lists
            left_audio.append(left)
            right_audio.append(right)
            # Update the previous frequency for the next step's potential transition
            previous_freq = audio_step.end_frequency
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
    sample_rate: int,
    left: np.ndarray,
    right: np.ndarray,
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

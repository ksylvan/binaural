#!/usr/bin/env python

"""Binaural Beat Generator

Generates binaural beat audio from a YAML configuration file.
"""

import argparse
import wave
import sys
from typing import List, Tuple

import numpy as np
import yaml

DEFAULT_SAMPLE_RATE = 44100
DEFAULT_BASE_FREQUENCY = 100
DEFAULT_OUTPUT_FILENAME = "binaural_beats.wav"


def generate_tone(
    duration_sec: float,
    base_freq: float,
    freq_diff_start: float,
    freq_diff_end: float,
    sample_rate: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generates stereo audio data for binaural beats."""
    num_samples = int(sample_rate * duration_sec)
    t = np.linspace(0, duration_sec, num_samples, endpoint=False)
    freq_diff = np.linspace(freq_diff_start, freq_diff_end, num_samples)

    left_channel = np.sin(2 * np.pi * base_freq * t)
    right_channel = np.sin(2 * np.pi * (base_freq + freq_diff) * t)

    return left_channel, right_channel


def load_yaml_config(path: str) -> dict:
    """Loads and validates YAML configuration."""
    try:
        with open(path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
        if not isinstance(config, dict) or "steps" not in config:
            raise ValueError("YAML must have a 'steps' key with a list of steps.")
        return config
    except FileNotFoundError:
        sys.exit(f"Error: Config file '{path}' not found.")
    except yaml.YAMLError as e:
        sys.exit(f"Error parsing YAML file '{path}': {e}")
    except ValueError as e:
        sys.exit(f"Configuration error: {e}")


def validate_step(step: dict, previous_freq: float) -> Tuple[str, float, float, float]:
    """Validates and extracts necessary fields from a step."""
    step_type = step.get("type")
    duration_min = step.get("duration")

    if step_type not in ("stable", "transition"):
        raise ValueError(
            f"Invalid step type '{step_type}'. Must be 'stable' or 'transition'."
        )
    if not isinstance(duration_min, (int, float)) or duration_min <= 0:
        raise ValueError("Step duration must be a positive number.")

    duration_sec = duration_min * 60

    if step_type == "stable":
        freq = step.get("frequency")
        if not isinstance(freq, (int, float)):
            raise ValueError("Stable step must specify a valid 'frequency'.")
        return step_type, duration_sec, freq, freq

    start_freq = step.get("start_frequency", previous_freq)
    end_freq = step.get("end_frequency")

    if not all(isinstance(f, (int, float)) for f in (start_freq, end_freq)):
        raise ValueError(
            "Transition step must specify valid 'start_frequency' and 'end_frequency'."
        )

    return step_type, duration_sec, start_freq, end_freq


def generate_audio_sequence(
    steps: List[dict], base_freq: float, sample_rate: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Generates the complete audio sequence."""
    left_audio, right_audio = [], []
    previous_freq = None

    for idx, step in enumerate(steps, start=1):
        try:
            step_type, duration_sec, freq_start, freq_end = validate_step(
                step, previous_freq
            )
            print(
                f"Generating step {idx}: {step_type}, "
                f"{freq_start}Hz -> {freq_end}Hz, duration {duration_sec / 60:.2f}min"
            )
            left, right = generate_tone(
                duration_sec, base_freq, freq_start, freq_end, sample_rate
            )
            left_audio.append(left)
            right_audio.append(right)
            previous_freq = freq_end
        except ValueError as e:
            sys.exit(f"Error in step {idx}: {e}")

    return np.concatenate(left_audio), np.concatenate(right_audio)


def save_wav_file(filename: str, left: np.ndarray, right: np.ndarray, sample_rate: int):
    """Saves stereo audio data to a WAV file."""
    stereo_audio = np.vstack((left, right)).T
    max_val = np.max(np.abs(stereo_audio))
    if max_val > 0:
        stereo_audio = (stereo_audio / max_val * 32767).astype(np.int16)

    try:
        with wave.open(filename, "wb") as wf:
            wf.setnchannels(2)  # pylint: disable=no-member
            wf.setsampwidth(2)  # pylint: disable=no-member
            wf.setframerate(sample_rate)  # pylint: disable=no-member
            wf.writeframes(stereo_audio.tobytes())  # pylint: disable=no-member
        print(f"Audio file '{filename}' created successfully.")
    except (wave.Error, OSError) as e:
        sys.exit(f"Error writing WAV file '{filename}': {e}")


def main(script_path: str, output_path: str = None):
    """Main function to generate binaural beats from a YAML script."""
    config = load_yaml_config(script_path)

    base_freq = config.get("base_frequency", DEFAULT_BASE_FREQUENCY)
    sample_rate = config.get("sample_rate", DEFAULT_SAMPLE_RATE)
    output_filename = output_path or config.get(
        "output_filename", DEFAULT_OUTPUT_FILENAME
    )

    left_channel, right_channel = generate_audio_sequence(
        config["steps"], base_freq, sample_rate
    )
    save_wav_file(output_filename, left_channel, right_channel, sample_rate)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate binaural beats audio from a YAML script."
    )
    parser.add_argument("script", help="Path to YAML script.")
    parser.add_argument("-o", "--output", help="Output WAV file path.")
    args = parser.parse_args()

    main(args.script, args.output)

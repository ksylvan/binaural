"""Generates stereo audio data for binaural beats with volume envelope."""

import os
from typing import Optional, Tuple

import numpy as np
import soundfile as sf

from binaural.constants import SUPPORTED_FORMATS
from binaural.fade import apply_fade
from binaural.types import AudioStep, Tone
from binaural.exceptions import AudioGenerationError, UnsupportedFormatError


def generate_tone(
    sample_rate: int, duration_sec: float, tone: Tone
) -> Tuple[np.ndarray, np.ndarray]:
    """Generates stereo audio data for binaural beats with fades."""
    num_samples = int(sample_rate * duration_sec)
    if num_samples == 0:
        return np.array([]), np.array([])

    t = np.linspace(0, duration_sec, num_samples, endpoint=False)
    freq_diff = np.linspace(tone.freq_diff_start, tone.freq_diff_end, num_samples)

    left_channel = apply_fade(
        np.sin(2 * np.pi * tone.base_freq * t),
        sample_rate,
        tone.fade_in_sec,
        tone.fade_out_sec,
    )
    right_channel = apply_fade(
        np.sin(2 * np.pi * (tone.base_freq + freq_diff) * t),
        sample_rate,
        tone.fade_in_sec,
        tone.fade_out_sec,
    )

    return left_channel, right_channel


def generate_audio_sequence(
    sample_rate: int, base_freq: float, steps: list[dict]
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Generates the complete audio sequence."""
    left_audio, right_audio = [], []
    previous_freq: Optional[float] = None
    total_duration_sec = 0.0

    for idx, step in enumerate(steps, start=1):
        audio_step = AudioStep(**step)

        if audio_step.type == "transition":
            if audio_step.start_frequency is None:
                if previous_freq is None:
                    raise AudioGenerationError(
                        f"Error in step {idx}: Transition step must specify 'start_frequency'."
                    )
                audio_step.start_frequency = previous_freq
        else:
            audio_step.start_frequency = audio_step.frequency

        total_duration_sec += audio_step.duration
        print(f"Generating step {idx}: {audio_step}")

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

        left_audio.append(left)
        right_audio.append(right)
        previous_freq = audio_step.end_frequency

    if not left_audio:
        return np.array([]), np.array([]), 0.0

    return np.concatenate(left_audio), np.concatenate(right_audio), total_duration_sec


def save_audio_file(
    filename: str,
    sample_rate: int,
    left: np.ndarray,
    right: np.ndarray,
    total_duration_sec: float,
) -> None:
    """Saves stereo audio data."""
    _, ext = os.path.splitext(filename)
    if ext.lower() not in SUPPORTED_FORMATS:
        raise UnsupportedFormatError(
            f"Unsupported format '{ext}'. Supported formats: {', '.join(SUPPORTED_FORMATS)}"
        )

    if left.size == 0 or right.size == 0:
        raise AudioGenerationError("No audio data generated.")

    stereo_audio = np.column_stack((left, right))

    output_dir = os.path.dirname(filename)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    try:
        sf.write(filename, stereo_audio, sample_rate, subtype="PCM_16")
        minutes, seconds = divmod(total_duration_sec, 60)
        print(
            f"Audio file '{filename}' created successfully. "
            f"Total duration: {int(minutes)}m {seconds:.2f}s."
        )
    except (sf.SoundFileError, RuntimeError, IOError) as e:
        raise AudioGenerationError(f"Error writing audio file '{filename}': {e}") from e

"""Generates stereo audio data for binaural beats with volume envelope."""

import os
import logging
from typing import Optional, Tuple

import numpy as np
import soundfile as sf

from binaural.constants import SUPPORTED_FORMATS
from binaural.fade import apply_fade
from binaural.data_types import AudioStep, FadeInfo, FrequencyRange, Tone
from binaural.exceptions import (
    AudioGenerationError,
    UnsupportedFormatError,
    ConfigurationError,
)

logger = logging.getLogger(__name__)


def config_step_to_audio_step(step: dict, previous_freq: float | None) -> AudioStep:
    """Converts a configuration step to an AudioStep object."""
    if "type" not in step:
        raise ConfigurationError("Step must contain 'type' key.")
    if "duration" not in step:
        raise ConfigurationError("Step must contain 'duration' key.")
    if step["type"] == "stable":
        if "frequency" not in step:
            raise ConfigurationError("Stable step must contain 'frequency' key.")
        return AudioStep(
            duration=step["duration"],
            fade=FadeInfo(
                fade_in_sec=step.get("fade_in", 0.0),
                fade_out_sec=step.get("fade_out", 0.0),
            ),
            freq=FrequencyRange(
                type="stable", start=step["frequency"], end=step["frequency"]
            ),
        )
    if step["type"] == "transition":
        if "end_frequency" not in step:
            raise ConfigurationError("Transition step must contain 'end_frequency'.")
        if previous_freq is None and "start_frequency" not in step:
            raise ConfigurationError(
                "Transition step must contain 'start_frequency' if no previous frequency set."
            )
        start_freq = step.get("start_frequency", previous_freq)
        if "start_frequency" not in step:
            logger.debug(
                "No start frequency provided. Using previous frequency: %s", start_freq
            )
        return AudioStep(
            duration=step["duration"],
            fade=FadeInfo(
                fade_in_sec=step.get("fade_in", 0.0),
                fade_out_sec=step.get("fade_out", 0.0),
            ),
            freq=FrequencyRange(
                type="transition",
                start=start_freq,
                end=step["end_frequency"],
            ),
        )

    raise ConfigurationError(
        f"Invalid step type '{step['type']}'. Must be 'stable' or 'transition'."
    )


def generate_audio_sequence(
    sample_rate: int, base_freq: float, steps: list[dict]
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Generates complete audio sequence from steps."""
    left_audio, right_audio = [], []
    previous_freq: Optional[float] = None
    total_duration_sec = 0.0

    if not steps:
        raise ConfigurationError("No steps defined in configuration.")

    for idx, step in enumerate(steps, start=1):
        audio_step = config_step_to_audio_step(step, previous_freq)
        total_duration_sec += audio_step.duration
        logger.debug("Generating step %s: %s", idx, audio_step)

        left, right = generate_tone(
            sample_rate,
            audio_step.duration,
            Tone(
                base_freq,
                audio_step.freq.start,
                audio_step.freq.end,
                audio_step.fade.fade_in_sec,
                audio_step.fade.fade_out_sec,
            ),
        )

        if left.size == 0 or right.size == 0:
            raise AudioGenerationError(f"Generated zero audio data for step {idx}.")

        left_audio.append(left)
        right_audio.append(right)
        previous_freq = audio_step.freq.end

    return np.concatenate(left_audio), np.concatenate(right_audio), total_duration_sec


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
        logger.info(
            "Audio file '%s' created successfully. Total duration: %dm %.2fs.",
            filename,
            int(minutes),
            seconds,
        )
    except (sf.SoundFileError, RuntimeError, IOError) as e:
        raise AudioGenerationError(f"Error writing audio file '{filename}': {e}") from e

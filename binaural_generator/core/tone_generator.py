"""
Generates stereo audio data for binaural beats
with volume envelope and optional background noise.
"""

import logging
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
import soundfile as sf

from binaural_generator.core.constants import SUPPORTED_FORMATS
from binaural_generator.core.data_types import (
    AudioStep,
    FadeInfo,
    FrequencyRange,
    NoiseConfig,
    Tone,
)
from binaural_generator.core.exceptions import (
    AudioGenerationError,
    ConfigurationError,
    UnsupportedFormatError,
)
from binaural_generator.core.fade import apply_fade
from binaural_generator.core.noise import NoiseFactory

logger = logging.getLogger(__name__)


def _process_stable_step(
    step: dict[str, Any], duration: float, fade_info: FadeInfo
) -> AudioStep:
    """Process a 'stable' type step and return an AudioStep."""
    if "frequency" not in step:
        raise ConfigurationError("Stable step must contain 'frequency' key.")
    freq = float(step["frequency"])
    freq_range = FrequencyRange(type="stable", start=freq, end=freq)
    return AudioStep(duration=duration, fade=fade_info, freq=freq_range)


def _process_transition_step(
    step: dict[str, Any],
    previous_freq: Optional[float],
    duration: float,
    fade_info: FadeInfo,
) -> AudioStep:
    """Process a 'transition' type step and return an AudioStep."""
    if "end_frequency" not in step:
        raise ConfigurationError("Transition step must contain 'end_frequency'.")
    end_freq = float(step["end_frequency"])

    if "start_frequency" in step:
        start_freq = float(step["start_frequency"])
    elif previous_freq is not None:
        start_freq = previous_freq
        logger.debug(
            "Using implicit start frequency from previous step: %.2f Hz", start_freq
        )
    else:
        raise ConfigurationError(
            "First transition step must explicitly define 'start_frequency' "
            "or follow another step."
        )

    freq_range = FrequencyRange(type="transition", start=start_freq, end=end_freq)
    return AudioStep(duration=duration, fade=fade_info, freq=freq_range)


def config_step_to_audio_step(
    step: dict[str, Any], previous_freq: Optional[float]
) -> AudioStep:
    """Converts a configuration step dictionary into a validated AudioStep object."""
    for key in ["type", "duration"]:
        if key not in step:
            raise ConfigurationError(f"Step dictionary must contain a f'{key}' key.")

    step_type = step["type"]
    duration = step["duration"]
    fade_in_sec = float(step.get("fade_in_duration", 0.0))
    fade_out_sec = float(step.get("fade_out_duration", 0.0))

    if not isinstance(duration, (int, float)) or duration <= 0:
        raise ConfigurationError(
            f"Step duration must be a positive number, got '{duration}'."
        )
    if fade_in_sec < 0 or fade_out_sec < 0:
        raise ConfigurationError("Fade durations cannot be negative.")
    if fade_in_sec + fade_out_sec > duration:
        raise ConfigurationError(
            f"Sum of fade-in ({fade_in_sec}s) and fade-out ({fade_out_sec}s) "
            f"cannot exceed step duration ({duration}s)."
        )

    fade_info = FadeInfo(fade_in_sec=fade_in_sec, fade_out_sec=fade_out_sec)

    try:
        if step_type == "stable":
            return _process_stable_step(step, duration, fade_info)
        if step_type == "transition":
            return _process_transition_step(step, previous_freq, duration, fade_info)
        raise ConfigurationError(
            f"Invalid step type '{step_type}'. Must be 'stable' or 'transition'."
        )
    except (ValueError, TypeError) as e:
        raise ConfigurationError(f"Invalid value in step configuration: {e}") from e


def generate_tone(
    sample_rate: int, duration_sec: float, tone: Tone
) -> Tuple[np.ndarray, np.ndarray]:
    """Generates stereo audio data for a single binaural beat tone segment."""
    num_samples = int(sample_rate * max(0, duration_sec))
    if num_samples <= 0:
        return np.array([]), np.array([])

    t = np.linspace(0, duration_sec, num_samples, endpoint=False)
    freq_diff = np.linspace(tone.freq_diff_start, tone.freq_diff_end, num_samples)

    phase_left = 2 * np.pi * tone.base_freq * t
    phase_right = 2 * np.pi * (tone.base_freq + freq_diff) * t

    left_channel_raw = np.sin(phase_left)
    right_channel_raw = np.sin(phase_right)

    left_channel = apply_fade(
        left_channel_raw, sample_rate, tone.fade_in_sec, tone.fade_out_sec
    )
    right_channel = apply_fade(
        right_channel_raw, sample_rate, tone.fade_in_sec, tone.fade_out_sec
    )

    return left_channel, right_channel


def _process_beat_step(
    idx: int,
    step_dict: dict[str, Any],
    sample_rate: int,
    base_freq: float,
    previous_freq: Optional[float],
    *,
    title: str = "Binaural Beat",
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Processes a single step dict, generates audio, returns segments and end freq."""
    audio_step = config_step_to_audio_step(step_dict, previous_freq)
    logger.debug("Generating beat segment for step %d: %s", idx, audio_step)
    tone = audio_step.to_tone(base_freq, title)
    left_segment, right_segment = generate_tone(sample_rate, audio_step.duration, tone)

    if audio_step.duration > 0 and (left_segment.size == 0 or right_segment.size == 0):
        logger.warning(
            "Generated zero audio data for step %d despite positive duration (%.4fs).",
            idx,
            audio_step.duration,
        )

    return left_segment, right_segment, audio_step.duration, audio_step.freq.end


def _iterate_beat_steps(
    sample_rate: int,
    base_freq: float,
    steps: list[dict[str, Any]],
    title: str = "Binaural Beat",
) -> iter:
    """Iterates through configuration steps, yielding processed beat segments."""
    previous_freq: Optional[float] = None
    for idx, step_dict in enumerate(steps, start=1):
        try:
            left_segment, right_segment, step_duration, end_freq = _process_beat_step(
                idx, step_dict, sample_rate, base_freq, previous_freq, title=title
            )
            previous_freq = end_freq
            yield left_segment, right_segment, step_duration
        except ConfigurationError as e:
            raise ConfigurationError(f"Error processing step {idx}: {e}") from e
        except AudioGenerationError:
            raise
        except Exception as e:
            raise AudioGenerationError(
                f"Unexpected error during step {idx} generation: {e}"
            ) from e


def _generate_beat_segments(
    sample_rate: int,
    base_freq: float,
    steps: list[dict[str, Any]],
    title: str = "Binaural Beat",
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Generates and concatenates all binaural beat segments from config steps."""
    segments = list(_iterate_beat_steps(sample_rate, base_freq, steps, title))
    if not segments:
        raise ConfigurationError("No steps defined or processed in configuration.")

    left_segments, right_segments, durations = zip(*segments)
    concatenated_left = np.concatenate(left_segments) if left_segments else np.array([])
    concatenated_right = (
        np.concatenate(right_segments) if right_segments else np.array([])
    )
    total_duration = sum(durations)
    return concatenated_left, concatenated_right, total_duration


def mix_beats_and_noise(  # Renamed to be public for use in parallel.py
    left_beats: np.ndarray,
    right_beats: np.ndarray,
    noise_signal: np.ndarray,
    noise_amplitude: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Mixes generated beat signals with a generated noise signal."""
    scaled_noise = noise_signal * noise_amplitude
    beat_scale_factor = 1.0 - noise_amplitude
    scaled_left_beats = left_beats * beat_scale_factor
    scaled_right_beats = right_beats * beat_scale_factor

    target_len = len(scaled_left_beats)
    if len(scaled_noise) != target_len:
        logger.warning(
            "Noise length (%d) differs from beat length (%d). Adjusting noise length.",
            len(scaled_noise),
            target_len,
        )
        if len(scaled_noise) > target_len:
            scaled_noise = scaled_noise[:target_len]
        else:
            padding = target_len - len(scaled_noise)
            scaled_noise = np.pad(scaled_noise, (0, padding), "constant")

    left_final = scaled_left_beats + scaled_noise
    right_final = scaled_right_beats + scaled_noise
    return left_final, right_final


def _generate_and_mix_noise(
    sample_rate: int,
    total_duration_sec: float,
    noise_config: NoiseConfig,
    left_channel_beats: np.ndarray,
    right_channel_beats: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generates background noise (if configured) and mixes it sequentially."""
    total_num_samples = int(sample_rate * total_duration_sec)
    if (
        noise_config.type == "none"
        or noise_config.amplitude <= 0
        or total_num_samples <= 0
    ):
        return left_channel_beats, right_channel_beats

    logger.info(
        "Generating '%s' noise with amplitude %.3f...",
        noise_config.type,
        noise_config.amplitude,
    )
    try:
        noise_strategy = NoiseFactory.get_strategy(noise_config.type)
        noise_signal = noise_strategy.generate(total_num_samples)
    except Exception as e:
        raise AudioGenerationError(
            f"Error generating '{noise_config.type}' noise: {e}"
        ) from e

    logger.info("Mixing noise with beat signals...")
    try:
        left_final, right_final = mix_beats_and_noise(  # Use the public function
            left_channel_beats,
            right_channel_beats,
            noise_signal,
            noise_config.amplitude,
        )
        logger.info("Noise mixed successfully.")
        return left_final, right_final
    except Exception as e:
        raise AudioGenerationError(f"Error mixing noise: {e}") from e


def generate_audio_sequence(
    sample_rate: int,
    base_freq: float,
    steps: list[dict[str, Any]],
    noise_config: NoiseConfig,
    title: str = "Binaural Beat",
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Generates the complete stereo audio sequence sequentially."""
    logger.info("Generating binaural beat sequence...")
    left_beats, right_beats, total_duration_sec = _generate_beat_segments(
        sample_rate, base_freq, steps, title
    )
    logger.info("Beat sequence generated (%.2f seconds).", total_duration_sec)

    left_final, right_final = _generate_and_mix_noise(
        sample_rate, total_duration_sec, noise_config, left_beats, right_beats
    )

    left_final = left_final.astype(np.float64)
    right_final = right_final.astype(np.float64)
    return left_final, right_final, total_duration_sec


def save_audio_file(
    filename: str,
    sample_rate: int,
    left: np.ndarray,
    right: np.ndarray,
    total_duration_sec: float,
) -> None:
    """Saves the generated stereo audio data to a WAV or FLAC file."""
    file_path = Path(filename)
    file_ext = file_path.suffix.lower()

    if file_ext not in SUPPORTED_FORMATS:
        format_list = ", ".join(SUPPORTED_FORMATS)
        raise UnsupportedFormatError(
            f"Unsupported format '{file_ext}'. Supported formats: {format_list}"
        )
    if left.size == 0 or right.size == 0:
        raise AudioGenerationError("Cannot save file: No audio data generated.")

    stereo_audio = np.column_stack((left, right))

    output_dir = file_path.parent
    if output_dir and not output_dir.exists():
        logger.info("Creating output directory: %s", output_dir)
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise AudioGenerationError(
                f"Failed to create output directory '{output_dir}': {e}"
            ) from e

    logger.info("Writing audio file to: %s", file_path)
    try:
        sf.write(str(file_path), stereo_audio, sample_rate, subtype="PCM_16")
        minutes, seconds = divmod(total_duration_sec, 60)
        logger.info(
            "Audio file '%s' (%s format, %d Hz) created. Duration: %dm %.2fs.",
            file_path.name,
            file_ext,
            sample_rate,
            int(minutes),
            seconds,
        )
    except (sf.SoundFileError, RuntimeError, IOError) as e:
        raise AudioGenerationError(
            f"Error writing audio file '{file_path}': {e}"
        ) from e

"""
Generates stereo audio data for binaural beats
with volume envelope and optional background noise.
"""

import os
import logging
from typing import Optional, Tuple

import numpy as np
import soundfile as sf

from binaural.constants import SUPPORTED_FORMATS
from binaural.fade import apply_fade
from binaural.data_types import AudioStep, FadeInfo, FrequencyRange, Tone, NoiseConfig
from binaural.exceptions import (
    AudioGenerationError,
    UnsupportedFormatError,
    ConfigurationError,
)

from binaural.noise import NoiseFactory

logger = logging.getLogger(__name__)


def config_step_to_audio_step(step: dict, previous_freq: Optional[float]) -> AudioStep:
    """Converts a configuration step dictionary into a validated AudioStep object.

    Args:
        step: The dictionary representing a single step from the YAML config.
        previous_freq: The ending frequency of the previous step, used for implicit
                       start frequency in transitions.

    Returns:
        An AudioStep object representing the validated step.

    Raises:
        ConfigurationError: If the step dictionary is missing required keys or has
                            invalid values.
    """

    for key in ["type", "duration"]:
        if key not in step:
            raise ConfigurationError(f"Step dictionary must contain a '{key}' key.")

    step_type = step["type"]
    duration = step["duration"]

    # Extract fade information, defaulting to 0
    fade_in_sec = step.get("fade_in_duration", 0.0)
    fade_out_sec = step.get("fade_out_duration", 0.0)

    # Validate fades against duration before creating objects
    if fade_in_sec + fade_out_sec > duration:
        raise ConfigurationError(
            f"Sum of fade-in ({fade_in_sec}s) and fade-out "
            f"({fade_out_sec}s) cannot exceed step duration ({duration}s)."
        )

    fade_info = FadeInfo(fade_in_sec=fade_in_sec, fade_out_sec=fade_out_sec)

    try:
        # Handle 'stable' type steps
        if step_type == "stable":
            if "frequency" not in step:
                raise ConfigurationError("Stable step must contain 'frequency' key.")
            freq = step["frequency"]
            freq_range = FrequencyRange(type="stable", start=freq, end=freq)
            return AudioStep(duration=duration, fade=fade_info, freq=freq_range)

        # Handle 'transition' type steps
        if step_type == "transition":
            if "end_frequency" not in step:
                raise ConfigurationError(
                    "Transition step must contain 'end_frequency'."
                )
            end_freq = step["end_frequency"]
            start_freq = step.get("start_frequency", previous_freq)

            if start_freq is None:
                # This happens if it's the first step and start_frequency is omitted
                raise ConfigurationError(
                    "First transition step must explicitly define 'start_frequency'."
                )

            if "start_frequency" not in step:
                logger.debug(
                    "Using implicit start frequency from previous step: %.2f Hz",
                    start_freq,
                )

            freq_range = FrequencyRange(
                type="transition", start=start_freq, end=end_freq
            )
            return AudioStep(duration=duration, fade=fade_info, freq=freq_range)

        # Handle invalid step types
        raise ConfigurationError(
            f"Invalid step type '{step_type}'. Must be 'stable' or 'transition'."
        )
    except (ValueError, TypeError) as e:
        # Catch validation errors from dataclasses (e.g., negative duration/frequency)
        raise ConfigurationError(f"Invalid value in step configuration: {e}") from e


def generate_tone(
    sample_rate: int, duration_sec: float, tone: Tone
) -> Tuple[np.ndarray, np.ndarray]:
    """Generates stereo audio data for a single binaural beat tone segment with fades.

    Args:
        sample_rate: The audio sample rate in Hz.
        duration_sec: The duration of the tone segment in seconds.
        tone: A Tone object containing frequency and fade parameters.

    Returns:
        A tuple containing the left and right channel audio data as numpy arrays.
    """
    # Calculate the number of samples required for this duration
    num_samples = int(sample_rate * duration_sec)
    if num_samples <= 0:
        # Return empty arrays if duration is effectively zero
        return np.array([]), np.array([])

    # Create a time vector for the samples
    t = np.linspace(0, duration_sec, num_samples, endpoint=False)

    # Calculate the instantaneous frequency difference (beat frequency)
    # Linearly interpolates between start and end difference if they differ
    freq_diff = np.linspace(tone.freq_diff_start, tone.freq_diff_end, num_samples)

    # Generate the sine wave for the left channel (base frequency)
    left_channel_raw = np.sin(2 * np.pi * tone.base_freq * t)
    # Generate the sine wave for the right channel (base + difference)
    right_channel_raw = np.sin(2 * np.pi * (tone.base_freq + freq_diff) * t)

    # Apply fade-in and fade-out envelopes to both channels
    left_channel = apply_fade(
        left_channel_raw, sample_rate, tone.fade_in_sec, tone.fade_out_sec
    )
    right_channel = apply_fade(
        right_channel_raw, sample_rate, tone.fade_in_sec, tone.fade_out_sec
    )

    return left_channel, right_channel


def _process_beat_step(
    idx: int,
    step_dict: dict,
    sample_rate: int,
    base_freq: float,
    previous_freq: Optional[float],
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Process a single beat step and generate audio for it.

    Returns:
        Tuple containing left segment, right segment, step duration, and end frequency.
    """
    # Convert the dictionary step to a validated AudioStep object
    audio_step = config_step_to_audio_step(step_dict, previous_freq)
    logger.debug("Generating beat segment for step %d: %s", idx, audio_step)

    # Generate the tone for the current step
    tone = Tone(
        base_freq=base_freq,
        freq_diff_start=audio_step.freq.start,
        freq_diff_end=audio_step.freq.end,
        fade_in_sec=audio_step.fade.fade_in_sec,
        fade_out_sec=audio_step.fade.fade_out_sec,
    )
    left_segment, right_segment = generate_tone(sample_rate, audio_step.duration, tone)

    # Basic check for generated data
    if left_segment.size == 0 or right_segment.size == 0:
        logger.warning(
            "Generated zero audio data for step %d (duration might be too small).",
            idx,
        )
        if audio_step.duration > 0:
            raise AudioGenerationError(
                f"Generated zero audio data for non-zero duration step {idx}."
            )

    return left_segment, right_segment, audio_step.duration, audio_step.freq.end


def _iterate_beat_steps(sample_rate: int, base_freq: float, steps: list[dict]) -> iter:
    """Yields processed beat segments from configuration steps.

    Yields:
        Tuple containing left segment, right segment, and step duration.
    """
    previous_freq: Optional[float] = None
    for idx, step_dict in enumerate(steps, start=1):
        try:
            left_segment, right_segment, step_duration, end_freq = _process_beat_step(
                idx, step_dict, sample_rate, base_freq, previous_freq
            )
        except (ConfigurationError, ValueError, TypeError) as e:
            raise ConfigurationError(f"Error processing step {idx}: {e}") from e
        except AudioGenerationError:
            raise
        except Exception as e:
            raise AudioGenerationError(
                f"Unexpected error during step {idx} generation: {e}"
            ) from e
        previous_freq = end_freq
        yield left_segment, right_segment, step_duration


def _generate_beat_segments(
    sample_rate: int, base_freq: float, steps: list[dict]
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Generates binaural beat segments from configuration steps.

    Returns:
        Tuple containing left channel, right channel, and total duration.
    """
    segments = list(_iterate_beat_steps(sample_rate, base_freq, steps))
    if not segments:
        raise ConfigurationError("No steps defined in configuration.")

    left_segments, right_segments, durations = zip(*segments)
    return (
        np.concatenate(left_segments),
        np.concatenate(right_segments),
        sum(durations),
    )


def _generate_and_mix_noise(
    sample_rate: int,
    total_duration_sec: float,
    noise_config: NoiseConfig,
    left_channel_beats: np.ndarray,
    right_channel_beats: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generates noise and mixes it with beat channels.

    Returns:
        Tuple containing final left and right channels with noise mixed.
    """
    total_num_samples = int(sample_rate * total_duration_sec)

    # If no noise needed, return the original channels
    if (
        noise_config.type == "none"
        or noise_config.amplitude <= 0
        or total_num_samples <= 0
    ):
        return left_channel_beats, right_channel_beats

    logger.info(
        "Generating '%s' noise with amplitude %.2f...",
        noise_config.type,
        noise_config.amplitude,
    )

    try:
        # Use the Strategy pattern to get the appropriate noise generator
        noise_strategy = NoiseFactory.get_strategy(noise_config.type)
        noise_signal = noise_strategy.generate(total_num_samples)

        # Scale noise and mix with beats
        noise_signal *= noise_config.amplitude
        beat_scale_factor = 1.0 - noise_config.amplitude

        left_final = left_channel_beats * beat_scale_factor + noise_signal
        right_final = right_channel_beats * beat_scale_factor + noise_signal

        return left_final, right_final

    except Exception as e:
        raise AudioGenerationError(f"Error generating or mixing noise: {e}") from e


def generate_audio_sequence(
    sample_rate: int,
    base_freq: float,
    steps: list[dict],
    noise_config: NoiseConfig,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Generates the complete stereo audio sequence based on the YAML steps,
    including optional background noise.

    Args:
        sample_rate: The audio sample rate in Hz.
        base_freq: The base carrier frequency in Hz.
        steps: A list of dictionaries, each representing an audio generation step.
        noise_config: A NoiseConfig object specifying background noise settings.

    Returns:
        A tuple containing:
        - left_channel: Numpy array for the final left audio channel.
        - right_channel: Numpy array for the final right audio channel.
        - total_duration_sec: The total duration of the generated audio in seconds.

    Raises:
        ConfigurationError: If the steps list is empty or contains invalid steps.
        AudioGenerationError: If errors occur during tone or noise generation.
    """
    # Generate the binaural beat segments
    left_beats, right_beats, total_duration_sec = _generate_beat_segments(
        sample_rate, base_freq, steps
    )

    # Generate and mix noise with the beat segments
    left_final, right_final = _generate_and_mix_noise(
        sample_rate, total_duration_sec, noise_config, left_beats, right_beats
    )

    # Ensure output arrays are float type (soundfile expects float or int)
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
    """Saves the generated stereo audio data to a WAV or FLAC file.

    Args:
        filename: The path to the output audio file. Extension determines format.
        sample_rate: The audio sample rate in Hz.
        left: Numpy array for the left audio channel.
        right: Numpy array for the right audio channel.
        total_duration_sec: The total duration of the audio in seconds (for logging).

    Raises:
        UnsupportedFormatError: If the filename extension is not .wav or .flac.
        AudioGenerationError: If the audio data is empty or file writing fails.
    """
    # Extract file extension to determine format and check support
    _, ext = os.path.splitext(filename)
    if ext.lower() not in SUPPORTED_FORMATS:
        format_list = ", ".join(SUPPORTED_FORMATS)
        raise UnsupportedFormatError(
            f"Unsupported format '{ext}'. Supported formats: {format_list}",
        )

    # Check if there is any audio data to save
    if left.size == 0 or right.size == 0:
        raise AudioGenerationError("Cannot save file: No audio data generated.")

    # Combine left and right channels into a stereo format (num_samples x 2)
    stereo_audio = np.column_stack((left, right))

    # Ensure the output directory exists, create it if necessary
    output_dir = os.path.dirname(filename)
    if output_dir:
        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError as e:
            raise AudioGenerationError(
                f"Failed to create output directory '{output_dir}': {e}"
            ) from e

    # Write the audio data to the file using soundfile
    try:
        # Using PCM_16 for broad compatibility
        sf.write(filename, stereo_audio, sample_rate, subtype="PCM_16")
        minutes, seconds = divmod(total_duration_sec, 60)
        logger.info(
            "Audio file '%s' (%s) created successfully. Total duration: %dm %.2fs.",
            filename,
            ext.lower(),
            int(minutes),
            seconds,
        )
    except (sf.SoundFileError, RuntimeError, IOError) as e:
        # Catch errors during file writing
        raise AudioGenerationError(f"Error writing audio file '{filename}': {e}") from e

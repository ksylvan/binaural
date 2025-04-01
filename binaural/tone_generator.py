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

# Import noise generation functions
from binaural.noise import (
    generate_white_noise,
    generate_pink_noise,
    generate_brown_noise,
)

logger = logging.getLogger(__name__)


def config_step_to_audio_step(step: dict, previous_freq: float | None) -> AudioStep:
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
    # Basic validation for required keys
    if "type" not in step:
        raise ConfigurationError("Step dictionary must contain a 'type' key.")
    if "duration" not in step:
        raise ConfigurationError("Step dictionary must contain a 'duration' key.")

    step_type = step["type"]
    duration = step["duration"]

    # Extract fade information, defaulting to 0
    fade_info = FadeInfo(
        fade_in_sec=step.get("fade_in_duration", 0.0),  # Corrected key name
        fade_out_sec=step.get("fade_out_duration", 0.0),  # Corrected key name
    )

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
                    "Transition step using implicit start frequency from previous step: %.2f Hz",
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
        global_config: The full loaded configuration dictionary.

    Returns:
        A tuple containing:
        - left_channel: Numpy array for the final left audio channel.
        - right_channel: Numpy array for the final right audio channel.
        - total_duration_sec: The total duration of the generated audio in seconds.

    Raises:
        ConfigurationError: If the steps list is empty or contains invalid steps.
        AudioGenerationError: If errors occur during tone or noise generation.
    """
    left_audio_segments, right_audio_segments = [], []
    previous_freq: Optional[float] = None
    total_duration_sec = 0.0

    # Ensure there are steps to process
    if not steps:
        raise ConfigurationError("No steps defined in configuration.")

    # --- Generate Binaural Beats Segments ---
    # Iterate through each step defined in the configuration
    for idx, step_dict in enumerate(steps, start=1):
        try:
            # Convert the dictionary step to a validated AudioStep object
            audio_step = config_step_to_audio_step(step_dict, previous_freq)
            total_duration_sec += audio_step.duration
            logger.debug("Generating beat segment for step %d: %s", idx, audio_step)

            # Generate the tone for the current step
            left_segment, right_segment = generate_tone(
                sample_rate,
                audio_step.duration,
                Tone(
                    base_freq=base_freq,
                    freq_diff_start=audio_step.freq.start,
                    freq_diff_end=audio_step.freq.end,
                    fade_in_sec=audio_step.fade.fade_in_sec,
                    fade_out_sec=audio_step.fade.fade_out_sec,
                ),
            )

            # Basic check for generated data
            if left_segment.size == 0 or right_segment.size == 0:
                # This case should ideally be handled by generate_tone returning empty arrays
                # but adding a check here for robustness.
                logger.warning(
                    "Generated zero audio data for step %d (duration might be too small).",
                    idx,
                )
                # Continue if duration was zero, raise if unexpected
                if audio_step.duration > 0:
                    raise AudioGenerationError(
                        f"Generated zero audio data for non-zero duration step {idx}."
                    )

            # Append the generated segments to the lists
            left_audio_segments.append(left_segment)
            right_audio_segments.append(right_segment)

            # Update the ending frequency for the next iteration
            previous_freq = audio_step.freq.end

        except (ConfigurationError, ValueError, TypeError) as e:
            # Catch configuration errors specific to this step
            raise ConfigurationError(f"Error processing step {idx}: {e}") from e
        except AudioGenerationError as e:
            # Re-raise audio generation errors
            raise e
        except Exception as e:
            # Catch unexpected errors during step processing
            raise AudioGenerationError(
                f"Unexpected error during step {idx} generation: {e}"
            ) from e

    # Concatenate all generated segments into single left and right channels
    left_channel_beats = (
        np.concatenate(left_audio_segments) if left_audio_segments else np.array([])
    )
    right_channel_beats = (
        np.concatenate(right_audio_segments) if right_audio_segments else np.array([])
    )

    # --- Generate and Mix Background Noise ---
    total_num_samples = int(sample_rate * total_duration_sec)
    noise_signal = np.zeros(total_num_samples)

    if (
        noise_config.type != "none"
        and noise_config.amplitude > 0
        and total_num_samples > 0
    ):
        logger.info(
            "Generating '%s' noise with amplitude %.2f...",
            noise_config.type,
            noise_config.amplitude,
        )
        try:
            # Select the appropriate noise generation function
            if noise_config.type == "white":
                noise_signal = generate_white_noise(total_num_samples)
            elif noise_config.type == "pink":
                noise_signal = generate_pink_noise(total_num_samples)
            elif noise_config.type == "brown":
                noise_signal = generate_brown_noise(total_num_samples)
            else:
                # This case should be prevented by NoiseConfig validation, but handle defensively
                logger.warning(
                    "Unsupported noise type '%s' specified.", noise_config.type
                )

            # Scale the noise by its configured amplitude
            noise_signal *= noise_config.amplitude

            # Mix noise with beats: scale both to prevent clipping
            # combined = beat_signal * (1 - noise_amplitude) + noise_signal
            # Note: noise_signal already includes amplitude scaling
            beat_scale_factor = 1.0 - noise_config.amplitude

            left_channel_final = left_channel_beats * beat_scale_factor + noise_signal
            right_channel_final = right_channel_beats * beat_scale_factor + noise_signal

            # Optional: Re-normalize final signal if concerned about minor clipping
            # max_amp = np.max(np.abs(np.concatenate((left_channel_final, right_channel_final))))
            # if max_amp > 1.0:
            #     logger.warning("Potential clipping detected after mixing noise. Re-normalizing.")
            #     left_channel_final /= max_amp
            #     right_channel_final /= max_amp

        except Exception as e:
            # Catch errors during noise generation or mixing
            raise AudioGenerationError(f"Error generating or mixing noise: {e}") from e
    else:
        # If no noise, the final channels are just the beat channels
        left_channel_final = left_channel_beats
        right_channel_final = right_channel_beats

    # Ensure output arrays are float type (soundfile expects float or int)
    left_channel_final = left_channel_final.astype(np.float64)
    right_channel_final = right_channel_final.astype(np.float64)

    return left_channel_final, right_channel_final, total_duration_sec


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
        raise UnsupportedFormatError(
            f"Unsupported format '{ext}'. Supported formats: {', '.join(SUPPORTED_FORMATS)}"
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

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
    # FrequencyRange validates non-negative frequency
    freq_range = FrequencyRange(type="stable", start=freq, end=freq)
    # AudioStep validates duration and fade sum
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

    # Determine start frequency: explicit, implicit from previous, or error
    if "start_frequency" in step:
        start_freq = float(step["start_frequency"])
    elif previous_freq is not None:
        start_freq = previous_freq
        logger.debug(
            "Using implicit start frequency from previous step: %.2f Hz",
            start_freq,
        )
    else:
        # This happens if it's the first step and start_frequency is omitted
        raise ConfigurationError(
            "First transition step must explicitly define 'start_frequency' "
            "or follow another step."
        )

    # FrequencyRange validates non-negative frequencies
    freq_range = FrequencyRange(type="transition", start=start_freq, end=end_freq)
    # AudioStep validates duration and fade sum
    return AudioStep(duration=duration, fade=fade_info, freq=freq_range)


def config_step_to_audio_step(
    step: dict[str, Any], previous_freq: Optional[float]
) -> AudioStep:
    """Converts a configuration step dictionary into a validated AudioStep object.

    Args:
        step: The dictionary representing a single step from the YAML config.
              Expected keys depend on 'type' ('stable' or 'transition').
        previous_freq: The ending frequency (Hz) of the previous step, used for implicit
                       start frequency in 'transition' steps if 'start_frequency'
                       is not provided.

    Returns:
        An AudioStep object representing the validated step.

    Raises:
        ConfigurationError: If the step dictionary is missing required keys, has
                            invalid values (e.g., type, duration, frequencies),
                            or if fades exceed duration.
    """
    # Check for mandatory keys 'type' and 'duration'
    for key in ["type", "duration"]:
        if key not in step:
            raise ConfigurationError(f"Step dictionary must contain a f'{key}' key.")

    step_type = step["type"]
    duration = step["duration"]

    # Extract fade information, defaulting to 0.0 if not present
    fade_in_sec = float(step.get("fade_in_duration", 0.0))
    fade_out_sec = float(step.get("fade_out_duration", 0.0))

    # Basic duration validation before creating FadeInfo
    if not isinstance(duration, (int, float)) or duration <= 0:
        raise ConfigurationError(
            f"Step duration must be a positive number, got '{duration}'."
        )

    # Validate fades against duration before creating objects
    if fade_in_sec < 0 or fade_out_sec < 0:
        raise ConfigurationError("Fade durations cannot be negative.")
    if fade_in_sec + fade_out_sec > duration:
        raise ConfigurationError(
            f"Sum of fade-in ({fade_in_sec}s) and fade-out "
            f"({fade_out_sec}s) cannot exceed step duration ({duration}s)."
        )

    # Create FadeInfo object (validation for negative fades is done above)
    fade_info = FadeInfo(fade_in_sec=fade_in_sec, fade_out_sec=fade_out_sec)

    try:
        # Dispatch to appropriate handler based on step type
        if step_type == "stable":
            return _process_stable_step(step, duration, fade_info)
        if step_type == "transition":
            return _process_transition_step(step, previous_freq, duration, fade_info)
        raise ConfigurationError(
            f"Invalid step type '{step_type}'. Must be 'stable' or 'transition'."
        )
    except (ValueError, TypeError) as e:
        # Catch validation errors from dataclasses or float conversion
        raise ConfigurationError(f"Invalid value in step configuration: {e}") from e


def generate_tone(
    sample_rate: int, duration_sec: float, tone: Tone
) -> Tuple[np.ndarray, np.ndarray]:
    """Generates stereo audio data for a single binaural beat tone segment.

    Applies linear frequency transition (if start/end frequencies differ) and
    volume fades (fade-in/fade-out).

    Args:
        sample_rate: The audio sample rate in Hz.
        duration_sec: The duration of the tone segment in seconds.
        tone: A Tone object containing base frequency,
            frequency differences (start/end), and fade durations.

    Returns:
        A tuple containing the left and right channel audio data as numpy arrays.
        Returns empty arrays if duration_sec is non-positive.
    """
    # Calculate the number of samples required for this duration
    # Use max(0, ...) to handle potential negative duration input gracefully
    num_samples = int(sample_rate * max(0, duration_sec))
    if num_samples <= 0:
        # Return empty arrays if duration is zero or negative
        return np.array([]), np.array([])

    # Create a time vector representing sample times from 0 to duration_sec
    t = np.linspace(0, duration_sec, num_samples, endpoint=False)

    # Calculate the instantaneous frequency difference (beat frequency)
    # Linearly interpolates from freq_diff_start to freq_diff_end over the duration
    freq_diff = np.linspace(tone.freq_diff_start, tone.freq_diff_end, num_samples)

    # --- Generate Sine Waves --- #
    # Argument for the sine function (phase)
    # Left channel: 2 * pi * base_frequency * time
    phase_left = 2 * np.pi * tone.base_freq * t
    # Right channel: 2 * pi * (base_frequency + frequency_difference) * time
    phase_right = 2 * np.pi * (tone.base_freq + freq_diff) * t

    # Generate the raw sine waves for left and right channels
    left_channel_raw = np.sin(phase_left)
    right_channel_raw = np.sin(phase_right)

    # --- Apply Fades --- #
    # Apply fade-in and fade-out envelopes using the apply_fade utility
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
    *,  # Add keyword-only separator
    title: str = "Binaural Beat",
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Processes a single step dict, generates audio, returns segments and end freq.

    Internal helper function for iterating through steps.

    Args:
        idx: The 1-based index of the current step (for logging/error messages).
        step_dict: The dictionary configuration for the current step.
        sample_rate: Audio sample rate in Hz.
        base_freq: Base carrier frequency in Hz.
        previous_freq: The ending frequency of the previous step (for transitions).
        title: The title for the audio session (for metadata).

    Returns:
        A tuple containing:
        - left_segment: Numpy array for the left channel audio segment.
        - right_segment: Numpy array for the right channel audio segment.
        - step_duration: The duration of this step in seconds.
        - end_freq: The binaural beat frequency at the end of this step.

    Raises:
        ConfigurationError: If step configuration is invalid.
        AudioGenerationError: If audio generation fails unexpectedly for a valid step.
    """
    # Convert the dictionary step to a validated AudioStep object
    # This handles structural and value validation for the step
    audio_step = config_step_to_audio_step(step_dict, previous_freq)
    logger.debug("Generating beat segment for step %d: %s", idx, audio_step)

    # Create a Tone object from the validated AudioStep
    tone = audio_step.to_tone(base_freq, title)

    # Generate the audio segments for this step
    left_segment, right_segment = generate_tone(sample_rate, audio_step.duration, tone)

    # Check if generated data is unexpectedly empty for a positive duration step
    if audio_step.duration > 0 and (left_segment.size == 0 or right_segment.size == 0):
        # This might indicate an issue if sample_rate * duration resulted in 0 samples
        logger.warning(
            "Generated zero audio data for step %d despite positive duration (%.4fs). "
            "Check if duration is extremely small relative to sample rate.",
            idx,
            audio_step.duration,
        )
        # Optionally, raise an error if this is considered critical
        # raise AudioGenerationError(
        #     f"Generated zero audio data for non-zero duration step {idx}."
        # )

    return left_segment, right_segment, audio_step.duration, audio_step.freq.end


def _iterate_beat_steps(
    sample_rate: int,
    base_freq: float,
    steps: list[dict[str, Any]],
    title: str = "Binaural Beat",
) -> iter:
    """Iterates through configuration steps, yielding processed beat segments.

    Internal helper that manages the state (previous_freq) between steps.

    Args:
        sample_rate: Audio sample rate in Hz.
        base_freq: Base carrier frequency in Hz.
        steps: List of step configuration dictionaries.
        title: The title of the audio session.

    Yields:
        Tuple containing (left_segment, right_segment, step_duration).

    Raises:
        ConfigurationError: If any step has invalid configuration.
        AudioGenerationError: If an unexpected error occurs during generation.
    """
    previous_freq: Optional[float] = None  # Track the end frequency of the last step
    for idx, step_dict in enumerate(steps, start=1):
        try:
            # Process the current step
            left_segment, right_segment, step_duration, end_freq = _process_beat_step(
                idx, step_dict, sample_rate, base_freq, previous_freq, title=title
            )
        except ConfigurationError as e:
            # Re-raise configuration errors with step context
            raise ConfigurationError(f"Error processing step {idx}: {e}") from e
        except AudioGenerationError:
            # Re-raise audio generation errors directly
            raise
        except Exception as e:
            # Wrap unexpected errors
            raise AudioGenerationError(
                f"Unexpected error during step {idx} generation: {e}"
            ) from e

        # Update the previous frequency for the next iteration
        previous_freq = end_freq
        # Yield the results for this step
        yield left_segment, right_segment, step_duration


def _generate_beat_segments(
    sample_rate: int,
    base_freq: float,
    steps: list[dict[str, Any]],
    title: str = "Binaural Beat",
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Generates and concatenates all binaural beat segments from config steps.

    Internal helper function.

    Args:
        sample_rate: Audio sample rate in Hz.
        base_freq: Base carrier frequency in Hz.
        steps: List of step configuration dictionaries.
        title: The title of the audio session.

    Returns:
        Tuple containing:
        - concatenated left channel numpy array.
        - concatenated right channel numpy array.
        - total duration in seconds.

    Raises:
        ConfigurationError: If the steps list is empty.
    """
    # Use the iterator to process all steps and collect the results
    # list() consumes the iterator defined in _iterate_beat_steps
    segments = list(_iterate_beat_steps(sample_rate, base_freq, steps, title))

    # Check if any segments were generated
    if not segments:
        raise ConfigurationError("No steps defined or processed in configuration.")

    # Unzip the collected segments into separate lists/tuples
    left_segments, right_segments, durations = zip(*segments)

    # Concatenate all segments for each channel and sum durations
    concatenated_left = np.concatenate(left_segments) if left_segments else np.array([])
    concatenated_right = (
        np.concatenate(right_segments) if right_segments else np.array([])
    )
    total_duration = sum(durations)

    return concatenated_left, concatenated_right, total_duration


def _mix_beats_and_noise(
    left_beats: np.ndarray,
    right_beats: np.ndarray,
    noise_signal: np.ndarray,
    noise_amplitude: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Mixes generated beat signals with a generated noise signal.

    Handles scaling of both signals and ensures length compatibility.

    Args:
        left_beats: The generated left channel beat signal.
        right_beats: The generated right channel beat signal.
        noise_signal: The generated noise signal (assumed not None).
        noise_amplitude: The amplitude (0.0 to 1.0) for the noise signal.

    Returns:
        Tuple of final mixed left and right channel numpy arrays.
    """
    # Scale the noise signal by its configured amplitude
    scaled_noise = noise_signal * noise_amplitude

    # Scale the beat signal down to make room for the noise
    beat_scale_factor = 1.0 - noise_amplitude
    scaled_left_beats = left_beats * beat_scale_factor
    scaled_right_beats = right_beats * beat_scale_factor

    # Ensure noise signal length matches concatenated beats length
    target_len = len(scaled_left_beats)
    if len(scaled_noise) != target_len:
        logger.warning(
            "Noise length (%d) differs from combined beat "
            "length (%d). Adjusting noise length.",
            len(scaled_noise),
            target_len,
        )
        if len(scaled_noise) > target_len:
            scaled_noise = scaled_noise[:target_len]  # Truncate
        else:
            padding = target_len - len(scaled_noise)  # Pad
            scaled_noise = np.pad(scaled_noise, (0, padding), "constant")

    # Add the scaled noise to the scaled beat signals
    left_final = scaled_left_beats + scaled_noise
    right_final = scaled_right_beats + scaled_noise

    # Optional: Clip final signal just in case of minor floating point overshoot
    # left_final = np.clip(left_final, -1.0, 1.0)
    # right_final = np.clip(right_final, -1.0, 1.0)

    return left_final, right_final


def _generate_and_mix_noise(
    sample_rate: int,
    total_duration_sec: float,
    noise_config: NoiseConfig,
    left_channel_beats: np.ndarray,
    right_channel_beats: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generates background noise (if configured) and mixes it with beat channels.

    This is the sequential version used by generate_audio_sequence.

    Args:
        sample_rate: Audio sample rate in Hz.
        total_duration_sec: Total duration of the audio in seconds.
        noise_config: NoiseConfig object specifying noise type and amplitude.
        left_channel_beats: Numpy array for the generated left beat channel.
        right_channel_beats: Numpy array for the generated right beat channel.

    Returns:
        Tuple containing the final left and right channel numpy arrays.

    Raises:
        AudioGenerationError: If noise generation or mixing fails.
    """
    total_num_samples = int(sample_rate * total_duration_sec)

    # Check if noise generation is needed
    if (
        noise_config.type == "none"
        or noise_config.amplitude <= 0
        or total_num_samples <= 0
    ):
        return left_channel_beats, right_channel_beats

    # --- Generate Noise --- #
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

    # Call the dedicated mixing function
    logger.info("Mixing noise with beat signals...")
    try:
        left_final, right_final = _mix_beats_and_noise(
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
    """Generates the complete stereo audio sequence based on the YAML steps,
    including optional background noise.

    Args:
        sample_rate: The audio sample rate in Hz.
        base_freq: The base carrier frequency in Hz.
        steps: A list of dictionaries, each representing an audio generation step
               as defined in the YAML configuration.
        noise_config: A NoiseConfig object specifying background noise settings.
        title: The title of the audio session.

    Returns:
        A tuple containing:
        - left_channel: Numpy array for the final left audio channel.
        - right_channel: Numpy array for the final right audio channel.
        - total_duration_sec: The total duration of the generated audio in seconds.

    Raises:
        ConfigurationError: If the steps list is empty or contains invalid steps.
        AudioGenerationError: If errors occur during tone or noise generation/mixing.
    """
    # --- Generate Binaural Beats --- #
    # Generate the concatenated beat sequence for left and right channels
    logger.info("Generating binaural beat sequence...")
    left_beats, right_beats, total_duration_sec = _generate_beat_segments(
        sample_rate, base_freq, steps, title
    )
    logger.info("Beat sequence generated (%.2f seconds).", total_duration_sec)

    # --- Generate and Mix Noise --- #
    # Mix the generated beats with background noise if configured
    left_final, right_final = _generate_and_mix_noise(
        sample_rate, total_duration_sec, noise_config, left_beats, right_beats
    )

    # --- Final Type Conversion --- #
    # Ensure output arrays are float64 for potentially higher precision,
    # although float32 might be sufficient and save memory.
    # soundfile handles both float32 and float64 when writing.
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
        filename: The path (string) to the output audio file.
                  The file extension (.wav or .flac) determines the format.
        sample_rate: The audio sample rate in Hz.
        left: Numpy array for the left audio channel.
        right: Numpy array for the right audio channel.
        total_duration_sec: The total duration of the audio in seconds (for logging).

    Raises:
        UnsupportedFormatError: If the filename extension is not .wav or .flac.
        AudioGenerationError: If the audio data is empty, the output directory cannot
                              be created, or file writing fails.
    """
    # Use pathlib for robust path manipulation
    file_path = Path(filename)
    file_ext = file_path.suffix.lower()

    # Check if the file extension corresponds to a supported format
    if file_ext not in SUPPORTED_FORMATS:
        format_list = ", ".join(SUPPORTED_FORMATS)
        raise UnsupportedFormatError(
            f"Unsupported format '{file_ext}'. Supported formats: {format_list}",
        )

    # Check if there is any audio data to save
    if left.size == 0 or right.size == 0:
        raise AudioGenerationError("Cannot save file: No audio data generated.")

    # Combine left and right channels into a stereo format (num_samples x 2)
    # Ensure the arrays are C-contiguous for column_stack if necessary,
    # though usually they are.
    stereo_audio = np.column_stack((left, right))

    # --- Ensure Output Directory Exists --- #
    output_dir = file_path.parent
    # Create the directory if it exists and is not the root directory
    if output_dir and not output_dir.exists():
        logger.info("Creating output directory: %s", output_dir)
        try:
            # Create parent directories as needed
            output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            # Catch errors during directory creation (e.g., permission issues)
            raise AudioGenerationError(
                f"Failed to create output directory '{output_dir}': {e}"
            ) from e

    # --- Write Audio File --- #
    logger.info("Writing audio file to: %s", file_path)
    try:
        # Use soundfile.write to save the stereo audio
        # subtype="PCM_16" uses 16-bit integers, providing good quality
        # and compatibility for both WAV and FLAC.
        sf.write(str(file_path), stereo_audio, sample_rate, subtype="PCM_16")

        # Log success message with duration
        minutes, seconds = divmod(total_duration_sec, 60)
        logger.info(
            "Audio file '%s' (%s format, %d Hz) created successfully. "
            "Total duration: %dm %.2fs.",
            file_path.name,
            file_ext,
            sample_rate,
            int(minutes),
            seconds,
        )
    except (sf.SoundFileError, RuntimeError, IOError) as e:
        # Catch potential errors during file writing (e.g., disk full, permissions)
        raise AudioGenerationError(
            f"Error writing audio file '{file_path}': {e}"
        ) from e

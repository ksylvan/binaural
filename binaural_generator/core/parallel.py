"""Parallel processing utilities for binaural beat generation."""

import concurrent.futures
import logging
from typing import Any, Optional, Tuple, List

import numpy as np

from binaural_generator.core.data_types import AudioStep, NoiseConfig
from binaural_generator.core.exceptions import AudioGenerationError, ConfigurationError
from binaural_generator.core.noise import NoiseFactory, NoiseStrategy
from binaural_generator.core.tone_generator import (
    # _generate_and_mix_noise, # Not used directly in parallel flow anymore
    _process_beat_step,
    config_step_to_audio_step,
    generate_tone,
)

logger = logging.getLogger(__name__)


def generate_step_in_parallel(
    idx: int,
    step_dict: dict[str, Any],
    sample_rate: int,
    base_freq: float,
    previous_freq: Optional[float],
    *,  # Keyword-only arguments separator
    title: str = "Binaural Beat",
) -> Tuple[int, np.ndarray, np.ndarray, float, float]:
    """Generate audio for a single step, to be used in parallel processing.

    This function adapts _process_beat_step for concurrent execution.

    Args:
        idx: The 1-based index of the current step (for ordering).
        step_dict: The dictionary configuration for the current step.
        sample_rate: Audio sample rate in Hz.
        base_freq: Base carrier frequency in Hz.
        previous_freq: The ending frequency of the previous step (for transitions).
        title: The title of the audio session.

    Returns:
        A tuple containing:
        - idx: The original index for maintaining sequence order
        - left_segment: Numpy array for the left channel audio segment.
        - right_segment: Numpy array for the right channel audio segment.
        - step_duration: The duration of this step in seconds.
        - end_freq: The binaural beat frequency at the end of this step.
    """
    # This remains the same, it processes a single step based on its config
    left_segment, right_segment, step_duration, end_freq = _process_beat_step(
        idx, step_dict, sample_rate, base_freq, previous_freq, title=title
    )
    return idx, left_segment, right_segment, step_duration, end_freq


def prepare_audio_steps(steps: list[dict[str, Any]]) -> list[AudioStep]:
    """Preprocess all steps to determine start frequencies for transition steps.

    This function resolves dependencies between steps by calculating all
    start frequencies upfront, enabling parallel generation later.

    Args:
        steps: List of step configuration dictionaries.

    Returns:
        List of validated AudioStep objects with all dependencies resolved.

    Raises:
        ConfigurationError: If steps list is empty or any step has invalid config.
    """
    if not steps:
        raise ConfigurationError("No steps provided in configuration.")

    audio_steps = []
    previous_freq = None

    # First pass: interpret all steps sequentially to resolve dependencies
    for idx, step_dict in enumerate(steps, start=1):
        try:
            audio_step = config_step_to_audio_step(step_dict, previous_freq)
            audio_steps.append(audio_step)
            # Store end frequency for next step
            previous_freq = audio_step.freq.end
        except ConfigurationError as e:
            raise ConfigurationError(f"Error processing step {idx}: {e}") from e
        except Exception as e:
            raise ConfigurationError(
                f"Unexpected error preparing step {idx}: {e}"
            ) from e

    return audio_steps


def _submit_tone_generation_tasks(
    executor: concurrent.futures.ThreadPoolExecutor,
    audio_steps: list[AudioStep],
    sample_rate: int,
    base_freq: float,
    *,  # Keyword-only arguments separator
    title: str = "Binaural Beat",
) -> list[Tuple[int, concurrent.futures.Future, float, float]]:
    """Submit tone generation tasks to the thread pool.

    Args:
        executor: The ThreadPoolExecutor to submit tasks to.
        audio_steps: Pre-processed AudioStep objects.
        sample_rate: The audio sample rate in Hz.
        base_freq: The base carrier frequency in Hz.
        title: The title of the audio session.

    Returns:
        List of tuples: (index, future, duration, end_frequency).
    """
    futures_context = []
    for idx, audio_step in enumerate(audio_steps, start=1):
        tone = audio_step.to_tone(base_freq, title)
        future = executor.submit(generate_tone, sample_rate, audio_step.duration, tone)
        futures_context.append((idx, future, audio_step.duration, audio_step.freq.end))
    return futures_context


def _submit_noise_task(
    executor: concurrent.futures.ThreadPoolExecutor,
    noise_config: NoiseConfig,
    total_num_samples: int,
) -> Optional[Tuple[concurrent.futures.Future, NoiseStrategy]]:
    """Submits the noise generation task if needed.

    Args:
        executor: The ThreadPoolExecutor.
        noise_config: Noise configuration.
        total_num_samples: Total samples required for the noise track.

    Returns:
        A tuple (future, noise_strategy) if noise task is submitted, else None.

    Raises:
        AudioGenerationError: If noise strategy lookup or task submission fails.
    """
    if (
        noise_config.type == "none"
        or noise_config.amplitude <= 0
        or total_num_samples <= 0
    ):
        return None  # No noise needed

    try:
        noise_strategy = NoiseFactory.get_strategy(noise_config.type)
        logger.info(
            "Submitting '%s' noise generation task (amplitude %.3f) for %d samples...",
            noise_config.type,
            noise_config.amplitude,
            total_num_samples,
        )
        noise_future = executor.submit(noise_strategy.generate, total_num_samples)
        return noise_future, noise_strategy
    except Exception as e:
        logger.error("Failed to submit noise generation task: %s", e, exc_info=True)
        raise AudioGenerationError(f"Noise generation setup failed: {e}") from e


def _collect_beat_results(
    beat_futures_with_context: list[
        Tuple[int, concurrent.futures.Future, float, float]
    ],
) -> list[Tuple[int, np.ndarray, np.ndarray, float, float]]:
    """Collect results from beat futures, wait for completion, sort by index.

    Args:
        beat_futures_with_context: List from _submit_tone_generation_tasks.

    Returns:
        List of sorted result tuples:
        (idx, left_segment, right_segment, duration, end_freq)

    Raises:
        AudioGenerationError: If any beat generation task failed.
    """
    results = []
    # Use as_completed for potentially better responsiveness if tasks
    # finish out of order. However, we need the original context (idx, duration,
    # end_freq) alongside the future.
    # Creating a map from future to context allows retrieval.
    future_to_context = {
        f: (idx, dur, endf) for idx, f, dur, endf in beat_futures_with_context
    }

    for future in concurrent.futures.as_completed(future_to_context):
        idx, duration, end_freq = future_to_context[future]
        try:
            left_seg, right_seg = future.result()
            results.append((idx, left_seg, right_seg, duration, end_freq))
        except Exception as e:
            # Raising immediately stops collection on first error
            raise AudioGenerationError(
                f"Error generating audio for step {idx}: {e}"
            ) from e

    # Sort results by original index to maintain sequence order
    results.sort(key=lambda x: x[0])
    logger.info("Beat segments generated and collected.")
    return results


def _collect_noise_result(
    noise_task: Optional[Tuple[concurrent.futures.Future, NoiseStrategy]],
    noise_config: NoiseConfig,  # For logging
) -> Optional[np.ndarray]:
    """Waits for and collects the noise generation result if the task was submitted.

    Args:
        noise_task: The tuple (future, strategy) from _submit_noise_task, or None.
        noise_config: Noise configuration (for logging).

    Returns:
        The generated noise signal as a numpy array, or None if no task was run/needed.

    Raises:
        AudioGenerationError: If the noise generation task failed.
    """
    if not noise_task:
        logger.debug("No noise task was submitted, skipping noise result collection.")
        return None

    noise_future, noise_strategy = noise_task
    try:
        # Using class name for logging type is clearer
        # than accessing internal config type again
        noise_type = noise_strategy.__class__.__name__.replace("Strategy", "")
        logger.info("Waiting for '%s' noise generation to complete...", noise_type)
        noise_signal = noise_future.result()  # Blocks until noise is done
        logger.info("'%s' noise generated successfully.", noise_type)
        return noise_signal
    except Exception as e:
        # This catches errors *during* the execution of noise_strategy.generate
        raise AudioGenerationError(
            f"Error during execution of '{noise_config.type}'"
            f"noise generation task: {e}"
        ) from e


def _combine_audio_segments(
    step_results: list[Tuple[int, np.ndarray, np.ndarray, float, float]],
) -> tuple[np.ndarray, np.ndarray, float]:
    """Combine audio segments into continuous channels.

    Args:
        step_results: List of sorted, processed audio segments with metadata.

    Returns:
        A tuple containing:
        - left_channel: Numpy array for the left audio channel.
        - right_channel: Numpy array for the right audio channel.
        - total_duration: The total duration in seconds.
    """
    if not step_results:
        logger.warning("No step results found to combine.")
        return np.array([]), np.array([]), 0.0

    # Extract sorted results - assumes results are pre-sorted by index
    try:
        _, left_segments, right_segments, durations, _ = zip(*step_results)
    except ValueError:
        logger.error("Failed to unpack step results. Data might be empty or malformed.")
        return np.array([]), np.array([]), 0.0

    # Concatenate all segments
    left_channel = np.concatenate(left_segments) if left_segments else np.array([])
    right_channel = np.concatenate(right_segments) if right_segments else np.array([])
    total_duration = sum(durations)

    logger.info(
        "Beat segments combined (Total duration from segments: %.2f seconds).",
        total_duration,
    )
    return left_channel, right_channel, total_duration


def _mix_noise_if_present(
    left_beats: np.ndarray,
    right_beats: np.ndarray,
    noise_signal: Optional[np.ndarray],
    noise_config: NoiseConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """Mixes the generated noise signal with the beat signals if noise is present.

    Args:
        left_beats: Concatenated left channel beat signal.
        right_beats: Concatenated right channel beat signal.
        noise_signal: The generated noise signal (or None).
        noise_config: Noise configuration settings.

    Returns:
        Tuple of final left and right channel numpy arrays.

    Raises:
        AudioGenerationError: If an error occurs during mixing.
    """
    if noise_signal is None or noise_config.amplitude <= 0:
        # No noise mixing needed
        if noise_config.type != "none" and noise_config.amplitude > 0:
            logger.info("Skipping noise mixing because noise signal was not generated.")
        else:
            logger.info("Skipping noise mixing (not configured or zero amplitude).")
        return left_beats, right_beats

    logger.info("Mixing '%s' noise with beat segments...", noise_config.type)
    try:
        # Scale the noise signal by its configured amplitude
        scaled_noise = noise_signal * noise_config.amplitude

        # Scale the beat signal down to make room for the noise
        beat_scale_factor = 1.0 - noise_config.amplitude
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

        logger.info("Noise mixed successfully.")
        return left_final, right_final

    except Exception as e:
        raise AudioGenerationError(f"Error mixing noise: {e}") from e


def generate_audio_sequence_parallel(
    sample_rate: int,
    base_freq: float,
    steps: list[dict[str, Any]],
    noise_config: NoiseConfig,
    *,  # Keyword-only arguments separator
    title: str = "Binaural Beat",
    max_workers: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Generates the complete stereo audio sequence in parallel, including noise.

    Args:
        sample_rate: The audio sample rate in Hz.
        base_freq: The base carrier frequency in Hz.
        steps: A list of dictionaries, each representing an audio generation step.
        noise_config: A NoiseConfig object specifying background noise settings.
        title: The title of the audio session.
        max_workers: Maximum number of worker threads. None uses CPU count.

    Returns:
        A tuple containing:
        - left_channel: Numpy array for the final left audio channel.
        - right_channel: Numpy array for the final right audio channel.
        - total_duration_sec: The total duration of the generated audio in seconds.

    Raises:
        ConfigurationError: If steps list is empty or contains invalid steps.
        AudioGenerationError: If errors occur during audio generation.
    """
    logger.info("Preparing audio steps for parallel generation...")
    audio_steps = prepare_audio_steps(steps)  # Validates and resolves dependencies

    # Calculate total duration and samples based on validated steps
    total_duration = sum(step.duration for step in audio_steps)
    total_num_samples = int(sample_rate * total_duration)
    logger.debug(
        "Total duration: %.2f s, Total samples: %d", total_duration, total_num_samples
    )

    # Variables to hold results from parallel execution
    noise_signal: Optional[np.ndarray] = None
    step_results: List[Tuple[int, np.ndarray, np.ndarray, float, float]] = (
        []
    )  # Type hint for clarity

    logger.info("Starting parallel generation of beats and noise...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks concurrently
        beat_futures_ctx = _submit_tone_generation_tasks(
            executor, audio_steps, sample_rate, base_freq, title=title
        )
        noise_task_ctx = _submit_noise_task(executor, noise_config, total_num_samples)

        # Collect results (waits for completion)
        # It's important to collect beats first; if noise fails,
        # we might still want the beats.
        step_results = _collect_beat_results(beat_futures_ctx)
        # Collect noise result - this will raise if the noise task
        # failed during execution
        noise_signal = _collect_noise_result(noise_task_ctx, noise_config)

    # --- Combine and Mix Sequentially (outside the executor block) ---
    left_beats, right_beats, combined_duration = _combine_audio_segments(step_results)

    # Verify combined duration against initial calculation
    if not np.isclose(combined_duration, total_duration):
        logger.warning(
            "Mismatch calculated duration (%.4f) vs combined "
            "segments (%.4f). Using calculated value.",
            total_duration,
            combined_duration,
        )
        # Prefer total_duration calculated initially from validated steps

    # Mix noise if it was successfully generated
    left_final, right_final = _mix_noise_if_present(
        left_beats, right_beats, noise_signal, noise_config
    )

    # Final type conversion ensure float64 as per original non-parallel logic
    left_final = left_final.astype(np.float64)
    right_final = right_final.astype(np.float64)

    # Return the final audio and the definitive total_duration
    return left_final, right_final, total_duration

"""Parallel processing utilities for binaural beat generation."""

import concurrent.futures
import logging
from typing import Any, Optional, Tuple

import numpy as np

from binaural_generator.core.data_types import AudioStep, NoiseConfig
from binaural_generator.core.exceptions import AudioGenerationError
from binaural_generator.core.noise import NoiseFactory
from binaural_generator.core.tone_generator import (
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
        ConfigurationError: If any step has invalid configuration.
    """
    audio_steps = []
    previous_freq = None

    # First pass: interpret all steps sequentially to resolve dependencies
    for _, step_dict in enumerate(steps, start=1):
        audio_step = config_step_to_audio_step(step_dict, previous_freq)
        audio_steps.append(audio_step)
        # Store end frequency for next step
        previous_freq = audio_step.freq.end

    return audio_steps


# The AudioStep class now has a to_tone method that replaces this function


def _submit_tone_generation_tasks(
    executor: concurrent.futures.ThreadPoolExecutor,
    steps: list[dict[str, Any]],
    audio_steps: list[AudioStep],
    sample_rate: int,
    base_freq: float,
    *,  # Keyword-only arguments separator
    title: str = "Binaural Beat",
) -> list:
    """Submit tone generation tasks to the thread pool.

    Args:
        executor: The ThreadPoolExecutor to submit tasks to
        steps: Original step configuration dictionaries
        audio_steps: Pre-processed AudioStep objects
        sample_rate: The audio sample rate in Hz
        base_freq: The base carrier frequency in Hz
        title: The title of the audio session

    Returns:
        List of futures with their context information
    """
    futures = []

    for idx, (_, audio_step) in enumerate(zip(steps, audio_steps), start=1):
        # Create tone from the step
        tone = audio_step.to_tone(base_freq, title)

        # Submit task to the executor
        future = executor.submit(generate_tone, sample_rate, audio_step.duration, tone)

        # Store future with its context
        futures.append((idx, future, audio_step.duration, audio_step.freq.end))

    return futures


def _collect_audio_results(
    futures: list,
) -> list:
    """Collect results from futures and format them for further processing.

    Args:
        futures: List of futures with their context information

    Returns:
        List of result tuples (idx, left_segment, right_segment, duration, end_freq)

    Raises:
        AudioGenerationError: If errors occur during result collection
    """
    results = []

    for idx, future, duration, end_freq in futures:
        try:
            left_seg, right_seg = future.result()
            results.append((idx, left_seg, right_seg, duration, end_freq))
        except Exception as e:
            raise AudioGenerationError(
                f"Error generating audio for step {idx}: {e}"
            ) from e

    # Sort results by original index to maintain sequence order
    results.sort(key=lambda x: x[0])
    return results


def _calculate_total_duration(step_results: list) -> float:
    """Calculate the total duration from step results.

    Args:
        step_results: List of processed audio segments with metadata

    Returns:
        Total duration in seconds
    """
    durations = [item[3] for item in step_results]
    return sum(durations)


def _generate_audio_segments_parallel(
    sample_rate: int,
    base_freq: float,
    steps: list[dict[str, Any]],
    audio_steps: list[AudioStep],
    *,  # Keyword-only arguments separator
    title: str = "Binaural Beat",
    max_workers: Optional[int] = None,
) -> tuple[list, float]:
    """Generate audio segments in parallel.

    Helper function to handle the parallel processing part of audio generation.

    Args:
        sample_rate: The audio sample rate in Hz.
        base_freq: The base carrier frequency in Hz.
        steps: A list of dictionaries, each representing an audio generation step.
        audio_steps: Pre-processed AudioStep objects with dependencies resolved.
        title: The title of the audio session.
        max_workers: Maximum number of worker threads. None uses CPU count.

    Returns:
        A tuple containing:
        - step_results: List of processed audio segments with metadata
        - total_duration: The total duration of all segments in seconds

    Raises:
        ConfigurationError: If steps list is empty or contains invalid steps.
        AudioGenerationError: If errors occur during audio generation.
    """
    logger.debug("Generating audio segments in parallel with %d workers.", max_workers)
    logger.debug("Steps to be generated: %s", steps)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tone generation tasks to the thread pool
        futures = _submit_tone_generation_tasks(
            executor, steps, audio_steps, sample_rate, base_freq, title=title
        )

        # Collect and process results
        step_results = _collect_audio_results(futures)

    # Calculate total duration
    total_duration = _calculate_total_duration(step_results)

    return step_results, total_duration


def _combine_audio_segments(
    step_results: list,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Combine audio segments into continuous channels.

    Helper function to extract and concatenate audio segments.

    Args:
        step_results: List of processed audio segments with metadata

    Returns:
        A tuple containing:
        - left_channel: Numpy array for the left audio channel
        - right_channel: Numpy array for the right audio channel
        - total_duration: The total duration in seconds
    """
    # Extract sorted results
    _, left_segments, right_segments, durations, _ = zip(*step_results)

    # Concatenate all segments
    left_channel = np.concatenate(left_segments) if left_segments else np.array([])
    right_channel = np.concatenate(right_segments) if right_segments else np.array([])
    total_duration = sum(durations)

    return left_channel, right_channel, total_duration


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

    This function parallelizes the audio generation process using multiple threads
    for faster execution on multi-core systems. It generates beat segments and
    the full noise track concurrently.

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

    # First pass: interpret all steps sequentially to resolve dependencies
    audio_steps = prepare_audio_steps(steps)

    # Calculate total duration and samples needed *before* parallel execution
    total_duration = sum(step.duration for step in audio_steps)
    total_num_samples = int(sample_rate * total_duration)
    logger.debug(
        "Total duration calculated: %.2f seconds (%d samples)",
        total_duration,
        total_num_samples,
    )

    noise_future = None
    noise_signal = None
    noise_strategy = None  # Keep strategy reference for logging type later

    logger.info("Generating audio segments and noise (if configured) in parallel...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit beat generation tasks
        beat_futures = _submit_tone_generation_tasks(
            executor, steps, audio_steps, sample_rate, base_freq, title=title
        )

        # Submit noise generation task if needed
        if (
            noise_config.type != "none"
            and noise_config.amplitude > 0
            and total_num_samples > 0
        ):
            try:
                noise_strategy = NoiseFactory.get_strategy(noise_config.type)
                logger.info(
                    "Submitting '%s' noise generation task"
                    " (amplitude %.3f) for %d samples...",
                    noise_config.type,
                    noise_config.amplitude,
                    total_num_samples,
                )
                noise_future = executor.submit(
                    noise_strategy.generate, total_num_samples
                )
            except Exception as e:
                logger.error(
                    "Failed to submit noise generation task: %s", e, exc_info=True
                )
                # Raise error to halt execution if noise generation setup fails
                raise AudioGenerationError(f"Noise generation setup failed: {e}") from e

        # Collect beat results (this inherently waits for beat futures to complete)
        step_results = _collect_audio_results(beat_futures)
        logger.info("Beat segments generated in parallel.")

        # Collect noise result (wait for noise future if it exists)
        if noise_future and noise_strategy:  # Check noise_strategy exists too
            try:
                logger.info(
                    "Waiting for '%s' noise generation to complete...",
                    noise_config.type,
                )
                noise_signal = noise_future.result()  # This blocks until noise is done
                logger.info("'%s' noise generated successfully.", noise_config.type)
            except Exception as e:
                # If noise generation itself fails after submission
                raise AudioGenerationError(
                    f"Error during execution of '{noise_config.type}'"
                    f" noise generation task: {e}"
                ) from e

    # --- Combine results ---
    # Combine beat segments (now happens after the parallel block)
    logger.debug("Combining beat segments...")
    left_beats, right_beats, calculated_total_duration = _combine_audio_segments(
        step_results
    )
    # Sanity check duration - use the duration calculated from audio_steps sum
    if not np.isclose(calculated_total_duration, total_duration):
        logger.warning(
            "Mismatch between summed step durations (%.4f) "
            "and combined segment duration (%.4f). Using summed value: %.4f.",
            total_duration,
            calculated_total_duration,
            total_duration,
        )
        # Stick with total_duration derived directly from audio_steps sum.

    logger.info(
        "Beat segments combined (Total calculated duration: %.2f seconds).",
        total_duration,
    )

    # Mix noise if it was successfully generated
    if noise_signal is not None and noise_config.amplitude > 0:
        logger.info("Mixing '%s' noise with beat segments...", noise_config.type)
        try:
            # --- Mix Noise and Beats ---
            # Scale the noise signal by its configured amplitude
            scaled_noise = noise_signal * noise_config.amplitude

            # Scale the beat signal down to make room for the noise
            beat_scale_factor = 1.0 - noise_config.amplitude
            scaled_left_beats = left_beats * beat_scale_factor
            scaled_right_beats = right_beats * beat_scale_factor

            # Ensure noise signal length matches concatenated beats length
            # This handles potential minor discrepancies from float calculations
            target_len = len(scaled_left_beats)
            if len(scaled_noise) != target_len:
                logger.warning(
                    "Noise length (%d) differs from "
                    "combined beat length (%d). Adjusting noise length.",
                    len(scaled_noise),
                    target_len,
                )
                if len(scaled_noise) > target_len:
                    # Truncate noise if longer
                    scaled_noise = scaled_noise[:target_len]
                else:
                    # Pad noise with zeros if shorter (less likely)
                    padding = target_len - len(scaled_noise)
                    scaled_noise = np.pad(scaled_noise, (0, padding), "constant")

            # Add the scaled noise to the scaled beat signals
            left_final = scaled_left_beats + scaled_noise
            right_final = scaled_right_beats + scaled_noise

            # Optional: Clip final signal just in case of minor floating point overshoot
            # left_final = np.clip(left_final, -1.0, 1.0)
            # right_final = np.clip(right_final, -1.0, 1.0)

            logger.info("Noise mixed successfully.")
        except Exception as e:
            # Catch errors during the mixing process
            raise AudioGenerationError(f"Error mixing noise: {e}") from e
    else:
        # No noise requested, amplitude is zero, or noise generation failed earlier
        if noise_config.type != "none" and noise_config.amplitude > 0:
            logger.info("Skipping noise mixing because noise signal was not generated.")
        else:
            logger.info("Skipping noise mixing (not configured or zero amplitude).")
        left_final = left_beats
        right_final = right_beats

    # Ensure output arrays are float64 for consistency
    left_final = left_final.astype(np.float64)
    right_final = right_final.astype(np.float64)

    return left_final, right_final, total_duration

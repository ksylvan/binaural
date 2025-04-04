"""Parallel processing utilities for binaural beat generation."""

import concurrent.futures
import logging
from typing import Any, Optional, Tuple

import numpy as np

from binaural.data_types import AudioStep, NoiseConfig
from binaural.exceptions import AudioGenerationError
from binaural.tone_generator import (_generate_and_mix_noise,
                                     _process_beat_step,
                                     config_step_to_audio_step, generate_tone)

logger = logging.getLogger(__name__)


def generate_step_in_parallel(
    idx: int,
    step_dict: dict[str, Any],
    sample_rate: int,
    base_freq: float,
    previous_freq: Optional[float],
) -> Tuple[int, np.ndarray, np.ndarray, float, float]:
    """Generate audio for a single step, to be used in parallel processing.

    This function adapts _process_beat_step for concurrent execution.

    Args:
        idx: The 1-based index of the current step (for ordering).
        step_dict: The dictionary configuration for the current step.
        sample_rate: Audio sample rate in Hz.
        base_freq: Base carrier frequency in Hz.
        previous_freq: The ending frequency of the previous step (for transitions).

    Returns:
        A tuple containing:
        - idx: The original index for maintaining sequence order
        - left_segment: Numpy array for the left channel audio segment.
        - right_segment: Numpy array for the right channel audio segment.
        - step_duration: The duration of this step in seconds.
        - end_freq: The binaural beat frequency at the end of this step.
    """
    left_segment, right_segment, step_duration, end_freq = _process_beat_step(
        idx, step_dict, sample_rate, base_freq, previous_freq
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
) -> list:
    """Submit tone generation tasks to the thread pool.

    Args:
        executor: The ThreadPoolExecutor to submit tasks to
        steps: Original step configuration dictionaries
        audio_steps: Pre-processed AudioStep objects
        sample_rate: The audio sample rate in Hz
        base_freq: The base carrier frequency in Hz

    Returns:
        List of futures with their context information
    """
    futures = []

    for idx, (_, audio_step) in enumerate(zip(steps, audio_steps), start=1):
        # Create tone from the step
        tone = audio_step.to_tone(base_freq)

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
    max_workers: Optional[int] = None,
) -> tuple[list, float]:
    """Generate audio segments in parallel.

    Helper function to handle the parallel processing part of audio generation.

    Args:
        sample_rate: The audio sample rate in Hz.
        base_freq: The base carrier frequency in Hz.
        steps: A list of dictionaries, each representing an audio generation step.
        audio_steps: Pre-processed AudioStep objects with dependencies resolved.
        max_workers: Maximum number of worker threads. None uses CPU count.

    Returns:
        A tuple containing:
        - step_results: List of processed audio segments with metadata
        - total_duration: The total duration of all segments in seconds

    Raises:
        ConfigurationError: If steps list is empty or contains invalid steps.
        AudioGenerationError: If errors occur during audio generation.
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tone generation tasks to the thread pool
        futures = _submit_tone_generation_tasks(
            executor, steps, audio_steps, sample_rate, base_freq
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
    max_workers: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Generates the complete stereo audio sequence in parallel.

    This function parallelizes the audio generation process using multiple threads
    for faster execution on multi-core systems.

    Args:
        sample_rate: The audio sample rate in Hz.
        base_freq: The base carrier frequency in Hz.
        steps: A list of dictionaries, each representing an audio generation step.
        noise_config: A NoiseConfig object specifying background noise settings.
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

    # First pass: interpret all steps to resolve dependencies
    audio_steps = prepare_audio_steps(steps)

    # Generate all steps in parallel
    logger.info("Generating audio steps in parallel...")

    # Generate audio segments in parallel
    step_results, total_duration = _generate_audio_segments_parallel(
        sample_rate, base_freq, steps, audio_steps, max_workers
    )

    # Combine the segments into continuous channels
    left_channel, right_channel, _ = _combine_audio_segments(step_results)

    logger.info("Audio segments generated in parallel (%.2f seconds).", total_duration)

    # Apply noise if configured
    left_final, right_final = _generate_and_mix_noise(
        sample_rate, total_duration, noise_config, left_channel, right_channel
    )

    # Ensure output arrays are float64 for consistency with non-parallel version
    left_final = left_final.astype(np.float64)
    right_final = right_final.astype(np.float64)

    return left_final, right_final, total_duration

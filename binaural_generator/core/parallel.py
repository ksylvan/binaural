"""Parallel processing utilities for binaural beat generation."""

import concurrent.futures
import logging
from typing import Any, Optional, Tuple, List

import numpy as np

from binaural_generator.core.data_types import AudioStep, NoiseConfig
from binaural_generator.core.exceptions import AudioGenerationError, ConfigurationError
from binaural_generator.core.noise import NoiseFactory, NoiseStrategy
from binaural_generator.core.tone_generator import (
    mix_beats_and_noise, # Import the public mixer
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
    *,
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
        ConfigurationError: If steps list is empty or any step has invalid config.
    """
    if not steps:
        raise ConfigurationError("No steps provided in configuration.")

    audio_steps = []
    previous_freq = None

    for idx, step_dict in enumerate(steps, start=1):
        try:
            audio_step = config_step_to_audio_step(step_dict, previous_freq)
            audio_steps.append(audio_step)
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
    *,
    title: str = "Binaural Beat",
) -> list[Tuple[int, concurrent.futures.Future, float, float]]:
    """Submit tone generation tasks to the thread pool."""
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
    """Submits the noise generation task if needed."""
    if (
        noise_config.type == "none"
        or noise_config.amplitude <= 0
        or total_num_samples <= 0
    ):
        return None

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
    """Collect results from beat futures, wait for completion, sort by index."""
    results = []
    future_to_context = {
        f: (idx, dur, endf) for idx, f, dur, endf in beat_futures_with_context
    }

    for future in concurrent.futures.as_completed(future_to_context):
        idx, duration, end_freq = future_to_context[future]
        try:
            left_seg, right_seg = future.result()
            results.append((idx, left_seg, right_seg, duration, end_freq))
        except Exception as e:
            raise AudioGenerationError(
                f"Error generating audio for step {idx}: {e}"
            ) from e

    results.sort(key=lambda x: x[0])
    logger.info("Beat segments generated and collected.")
    return results


def _collect_noise_result(
    noise_task: Optional[Tuple[concurrent.futures.Future, NoiseStrategy]],
    noise_config: NoiseConfig,
) -> Optional[np.ndarray]:
    """Waits for and collects the noise generation result if the task was submitted."""
    if not noise_task:
        logger.debug("No noise task submitted, skipping noise result collection.")
        return None

    noise_future, noise_strategy = noise_task
    try:
        noise_type = noise_strategy.__class__.__name__.replace("Strategy", "")
        logger.info("Waiting for '%s' noise generation to complete...", noise_type)
        noise_signal = noise_future.result()
        logger.info("'%s' noise generated successfully.", noise_type)
        return noise_signal
    except Exception as e:
        raise AudioGenerationError(
            f"Error during execution of '{noise_config.type}' noise task: {e}"
        ) from e


def _combine_audio_segments(
    step_results: list[Tuple[int, np.ndarray, np.ndarray, float, float]],
) -> tuple[np.ndarray, np.ndarray, float]:
    """Combine audio segments into continuous channels."""
    if not step_results:
        logger.warning("No step results found to combine.")
        return np.array([]), np.array([]), 0.0

    try:
        _, left_segments, right_segments, durations, _ = zip(*step_results)
    except ValueError:
        logger.error("Failed to unpack step results. Data might be empty or malformed.")
        return np.array([]), np.array([]), 0.0

    left_channel = np.concatenate(left_segments) if left_segments else np.array([])
    right_channel = np.concatenate(right_segments) if right_segments else np.array([])
    total_duration = sum(durations)

    logger.info(
        "Beat segments combined (Duration from segments: %.2f seconds).", total_duration
    )
    return left_channel, right_channel, total_duration


# Removed _mix_noise_if_present as mix_beats_and_noise is now imported and used directly

def _execute_parallel_tasks(
    audio_steps: list[AudioStep],
    noise_config: NoiseConfig,
    sample_rate: int,
    base_freq: float,
    total_num_samples: int,
    title: str,
    max_workers: Optional[int],
) -> Tuple[
    List[Tuple[int, np.ndarray, np.ndarray, float, float]], Optional[np.ndarray]
]:
    """Executes beat and noise generation tasks in parallel using a thread pool."""
    logger.info("Starting parallel generation of beats and noise...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        beat_futures_ctx = _submit_tone_generation_tasks(
            executor, audio_steps, sample_rate, base_freq, title=title
        )
        noise_task_ctx = _submit_noise_task(executor, noise_config, total_num_samples)

        # Wait for completion and collect results
        step_results = _collect_beat_results(beat_futures_ctx)
        noise_signal = _collect_noise_result(noise_task_ctx, noise_config)

    return step_results, noise_signal


def generate_audio_sequence_parallel(
    sample_rate: int,
    base_freq: float,
    steps: list[dict[str, Any]],
    noise_config: NoiseConfig,
    *,
    title: str = "Binaural Beat",
    max_workers: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Generates the complete stereo audio sequence in parallel, including noise."""
    logger.info("Preparing audio steps for parallel generation...")
    audio_steps = prepare_audio_steps(steps)

    total_duration = sum(step.duration for step in audio_steps)
    total_num_samples = int(sample_rate * total_duration)
    logger.debug(
        "Total duration: %.2f s, Total samples: %d", total_duration, total_num_samples
    )

    # Execute tasks in parallel
    step_results, noise_signal = _execute_parallel_tasks(
        audio_steps,
        noise_config,
        sample_rate,
        base_freq,
        total_num_samples,
        title,
        max_workers,
    )

    # Combine and Mix Sequentially
    left_beats, right_beats, combined_duration = _combine_audio_segments(step_results)

    # Verify combined duration (logging only)
    if not np.isclose(combined_duration, total_duration):
        logger.warning(
            "Mismatch calculated duration (%.4f) vs combined segments (%.4f).",
            total_duration,
            combined_duration,
        )

    # Mix noise if applicable
    if noise_signal is not None and noise_config.amplitude > 0:
        logger.info("Mixing '%s' noise with beat segments...", noise_config.type)
        try:
            left_final, right_final = mix_beats_and_noise(
                left_beats, right_beats, noise_signal, noise_config.amplitude
            )
        except Exception as e:
            raise AudioGenerationError(f"Error mixing noise: {e}") from e
    else:
        logger.info("Skipping noise mixing (not generated or zero amplitude).")
        left_final, right_final = left_beats, right_beats

    # Final type conversion
    left_final = left_final.astype(np.float64)
    right_final = right_final.astype(np.float64)

    return left_final, right_final, total_duration # Use initial total_duration

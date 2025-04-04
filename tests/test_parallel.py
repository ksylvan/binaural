"""Tests for the parallel processing utilities."""

import time
import pytest
import numpy as np

from binaural.data_types import AudioStep, NoiseConfig
from binaural.parallel import (
    prepare_audio_steps,
    generate_audio_sequence_parallel,
)
from binaural.tone_generator import generate_audio_sequence


@pytest.fixture(name="sample_steps")
def sample_steps_fixture():
    """Sample steps for testing."""
    return [
        {
            "type": "stable",
            "duration": 50.0,
            "frequency": 10.0,
            "fade_in_duration": 0.5,
            "fade_out_duration": 0.5,
        },
        {
            "type": "transition",
            "duration": 100.0,
            "end_frequency": 5.0,
        },
        {
            "type": "stable",
            "duration": 50.0,
            "frequency": 5.0,
        },
    ]


def test_prepare_audio_steps(sample_steps):
    """Test preparing audio steps for parallel processing."""
    audio_steps = prepare_audio_steps(sample_steps)

    # Check we have the right number of steps
    assert len(audio_steps) == 3

    # Check the steps have the correct type
    assert isinstance(audio_steps[0], AudioStep)
    assert isinstance(audio_steps[1], AudioStep)
    assert isinstance(audio_steps[2], AudioStep)

    # Check the frequencies are correctly resolved
    assert audio_steps[0].freq.start == 10.0
    assert audio_steps[0].freq.end == 10.0

    # Second step should have start frequency equal to end frequency of first step
    assert audio_steps[1].freq.start == 10.0
    assert audio_steps[1].freq.end == 5.0

    # Third step should have frequencies as specified
    assert audio_steps[2].freq.start == 5.0
    assert audio_steps[2].freq.end == 5.0


def test_parallel_vs_sequential_output(sample_steps):
    """Test that parallel and sequential processing produce the same output."""
    sample_rate = 44100
    base_freq = 200.0
    noise_config = NoiseConfig()

    # Generate audio using sequential processing
    left_seq, right_seq, duration_seq = generate_audio_sequence(
        sample_rate=sample_rate,
        base_freq=base_freq,
        steps=sample_steps,
        noise_config=noise_config,
    )

    # Generate audio using parallel processing
    left_par, right_par, duration_par = generate_audio_sequence_parallel(
        sample_rate=sample_rate,
        base_freq=base_freq,
        steps=sample_steps,
        noise_config=noise_config,
    )

    # Check durations are the same
    assert duration_seq == duration_par

    # Check audio array shapes are the same
    assert left_seq.shape == left_par.shape
    assert right_seq.shape == right_par.shape

    # Check audio content is the same (within floating-point precision)
    np.testing.assert_allclose(left_seq, left_par)
    np.testing.assert_allclose(right_seq, right_par)


@pytest.mark.performance
def test_performance_improvement(sample_steps, request: pytest.FixtureRequest):
    """Test that parallel processing is faster than sequential processing.

    This test is marked as performance because it involves actual timing comparisons.
    Use --run-performance flag to run this test.
    """
    if not request.config.getoption("--run-performance"):
        pytest.skip("Skipping performance test (use --run-performance to run)")

    # Create a larger list of steps for better performance testing
    large_steps = sample_steps * 10  # 30 steps total

    sample_rate = 44100
    base_freq = 200.0
    noise_config = NoiseConfig()

    # Time sequential processing
    start_seq = time.time()
    generate_audio_sequence(
        sample_rate=sample_rate,
        base_freq=base_freq,
        steps=large_steps,
        noise_config=noise_config,
    )
    seq_time = time.time() - start_seq

    # Time parallel processing
    start_par = time.time()
    generate_audio_sequence_parallel(
        sample_rate=sample_rate,
        base_freq=base_freq,
        steps=large_steps,
        noise_config=noise_config,
    )
    par_time = time.time() - start_par

    # Assert that parallel processing is faster (allow small margin for overhead)
    # Parallel should at least be 30% faster on multi-core systems
    assert par_time < seq_time * 0.7, (
        f"Parallel: {par_time:.2f}s, Sequential: {seq_time:.2f}s - "
        f"Not enough speedup ({(seq_time - par_time) / seq_time * 100:.1f}%)"
    )

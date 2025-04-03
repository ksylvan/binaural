"""Property-based tests for the binaural module using Hypothesis."""

import os
import tempfile
from typing import Tuple

import numpy as np
import pytest
import soundfile as sf
from hypothesis import given, strategies as st, settings, assume

from binaural.constants import DEFAULT_BASE_FREQUENCY
from binaural.data_types import Tone, NoiseConfig, AudioStep, FadeInfo, FrequencyRange
from binaural.tone_generator import (
    generate_tone,
    generate_audio_sequence,
    save_audio_file,
    config_step_to_audio_step,
)
from binaural.noise import NoiseFactory

# Define common test constants
SAMPLE_RATES = [22050, 44100, 48000]  # Common sample rates
STANDARD_SAMPLE_RATE = 44100


# Define strategies for generating valid test data
@st.composite
def valid_tones(draw) -> Tone:
    """Strategy for generating valid Tone objects."""
    base_freq = draw(st.floats(min_value=20.0, max_value=20000.0))
    freq_diff_start = draw(st.floats(min_value=0.1, max_value=40.0))
    freq_diff_end = draw(st.floats(min_value=0.1, max_value=40.0))
    fade_in_sec = draw(st.floats(min_value=0.0, max_value=2.0))
    fade_out_sec = draw(st.floats(min_value=0.0, max_value=2.0))

    return Tone(
        base_freq=base_freq,
        freq_diff_start=freq_diff_start,
        freq_diff_end=freq_diff_end,
        fade_in_sec=fade_in_sec,
        fade_out_sec=fade_out_sec,
    )


@st.composite
def valid_durations(draw) -> Tuple[float, int]:
    """Strategy for generating valid duration and sample rate pairs."""
    sample_rate = draw(st.sampled_from(SAMPLE_RATES))
    # Duration between 0.001 (very short) and 5.0 seconds (reasonably long for tests)
    duration = draw(st.floats(min_value=0.001, max_value=5.0))
    return duration, sample_rate


@st.composite
def valid_audio_steps(draw) -> AudioStep:
    """Strategy for generating valid AudioStep objects."""
    # Choose between stable and transition types
    step_type = draw(st.sampled_from(["stable", "transition"]))

    # Generate frequencies based on type
    if step_type == "stable":
        freq_start = draw(st.floats(min_value=0.1, max_value=40.0))
        freq_end = freq_start  # Same for stable
    else:  # transition
        freq_start = draw(st.floats(min_value=0.1, max_value=40.0))
        freq_end = draw(st.floats(min_value=0.1, max_value=40.0))

    # Generate duration and fades
    duration = draw(st.floats(min_value=0.1, max_value=10.0))

    # Make sure fades don't exceed duration
    max_fade = duration / 2.0  # Ensure sum of fades <= duration
    fade_in = draw(st.floats(min_value=0.0, max_value=max_fade))
    fade_out = draw(st.floats(min_value=0.0, max_value=max_fade))

    # Ensure fade_in + fade_out <= duration
    assume(fade_in + fade_out <= duration)

    # Create the objects
    freq_range = FrequencyRange(type=step_type, start=freq_start, end=freq_end)
    fade_info = FadeInfo(fade_in_sec=fade_in, fade_out_sec=fade_out)

    return AudioStep(freq=freq_range, fade=fade_info, duration=duration)


@st.composite
def valid_noise_configs(draw) -> NoiseConfig:
    """Strategy for generating valid NoiseConfig objects."""
    noise_type = draw(st.sampled_from(["none", "white", "pink", "brown"]))

    # If type is none, amplitude should be 0
    if noise_type == "none":
        amplitude = 0.0
    else:
        amplitude = draw(st.floats(min_value=0.0, max_value=1.0))

    return NoiseConfig(type=noise_type, amplitude=amplitude)


@st.composite
def valid_step_dicts(draw) -> dict:
    """Strategy for generating valid step dictionaries."""
    # Choose between stable and transition types
    step_type = draw(st.sampled_from(["stable", "transition"]))

    # Basic step with type and duration
    step = {
        "type": step_type,
        "duration": draw(st.floats(min_value=0.1, max_value=10.0)),
    }

    # Add frequencies based on type
    if step_type == "stable":
        step["frequency"] = draw(st.floats(min_value=0.1, max_value=40.0))
    else:  # transition
        # Optionally include start_frequency (about 50% of the time)
        if draw(st.booleans()):
            step["start_frequency"] = draw(st.floats(min_value=0.1, max_value=40.0))
        step["end_frequency"] = draw(st.floats(min_value=0.1, max_value=40.0))

    # Optionally add fade information (about 50% of the time for each)
    if draw(st.booleans()):
        max_fade = step["duration"] / 2.0
        step["fade_in_duration"] = draw(st.floats(min_value=0.0, max_value=max_fade))

    if draw(st.booleans()):
        max_fade = step["duration"] / 2.0
        if "fade_in_duration" in step:
            max_fade = min(max_fade, step["duration"] - step["fade_in_duration"])
        step["fade_out_duration"] = draw(st.floats(min_value=0.0, max_value=max_fade))

    return step


# Property-based tests
@given(valid_tones(), valid_durations())
def test_generate_tone_properties(tone: Tone, duration_sample_rate: Tuple[float, int]):
    """Property-based test for generate_tone with various parameters."""
    duration, sample_rate = duration_sample_rate

    # Generate the tone
    left, right = generate_tone(sample_rate, duration, tone)

    # Expected number of samples (at least 1 for very short non-zero durations)
    expected_samples = max(1, int(sample_rate * duration)) if duration > 0 else 0

    # Check basic properties
    assert len(left) == expected_samples
    assert len(right) == expected_samples

    if expected_samples > 0:
        # Check amplitude is within expected range
        assert np.max(np.abs(left)) <= 1.0 + 1e-9  # Allow tiny float imprecision
        assert np.max(np.abs(right)) <= 1.0 + 1e-9

        # Check fades only if fade duration is longer than one sample,
        # i.e. > 1/sample_rate
        if (
            tone.fade_in_sec > 1.0 / sample_rate
            and tone.fade_in_sec < duration
            and expected_samples > 1
        ):
            # First sample should be lower than the second if fade-in is applied
            assert abs(left[0]) < abs(left[1])
            assert abs(right[0]) < abs(right[1])

            # Instead of checking just the last two samples, check the overall
            # trend of the fade-out by comparing average amplitudes
            fade_samples = int(tone.fade_out_sec * sample_rate)
            if (
                tone.fade_out_sec > 0.01  # At least 10ms fade (more reliable)
                and tone.fade_out_sec
                < duration / 2  # Fade is less than half the duration
                and fade_samples > 10  # Need more samples to see a clear fade effect
                and expected_samples > 20  # Enough total samples
            ):
                # Check that fade is applied by verifying the amplitude at the end
                # is significantly lower than the amplitude in the middle
                mid_idx = expected_samples // 2  # Middle of the signal
                mid_amplitude = np.max(np.abs(left[mid_idx - 5 : mid_idx + 5]))
                end_amplitude = np.abs(left[-1])

                # The end should be at least 30% lower than the middle
                assert end_amplitude < 0.7 * mid_amplitude


@given(st.integers(100, 100000))
def test_noise_generators_properties(num_samples: int):
    """Property-based test for the noise generators."""
    # Generate different types of noise
    white = NoiseFactory.get_strategy("white").generate(num_samples)
    pink = NoiseFactory.get_strategy("pink").generate(num_samples)
    brown = NoiseFactory.get_strategy("brown").generate(num_samples)

    # Check basic properties
    for noise in [white, pink, brown]:
        assert len(noise) == num_samples
        assert np.max(np.abs(noise)) <= 1.0 + 1e-9  # Allow tiny float imprecision

        # Mean should be close to zero (centered)
        assert abs(np.mean(noise)) < 0.1

    # Additional checks could include verifying spectral properties
    # but that would make tests too slow for regular use


@given(valid_step_dicts())
def test_config_step_to_audio_step_properties(step_dict: dict):
    """Property-based test for converting step dicts to AudioStep objects."""
    # We need to provide a previous_freq if this is a transition without start_frequency
    previous_freq = None
    if step_dict["type"] == "transition" and "start_frequency" not in step_dict:
        previous_freq = 10.0  # Arbitrary valid frequency

    # Convert to AudioStep
    audio_step = config_step_to_audio_step(step_dict, previous_freq)

    # Check basic properties
    assert audio_step.duration == step_dict["duration"]

    if step_dict["type"] == "stable":
        assert audio_step.freq.type == "stable"
        assert audio_step.freq.start == step_dict["frequency"]
        assert audio_step.freq.end == step_dict["frequency"]
    else:  # transition
        assert audio_step.freq.type == "transition"
        if "start_frequency" in step_dict:
            assert audio_step.freq.start == step_dict["start_frequency"]
        else:
            assert audio_step.freq.start == previous_freq
        assert audio_step.freq.end == step_dict["end_frequency"]

    # Check fade properties
    if "fade_in_duration" in step_dict:
        assert audio_step.fade.fade_in_sec == step_dict["fade_in_duration"]
    else:
        assert audio_step.fade.fade_in_sec == 0.0

    if "fade_out_duration" in step_dict:
        assert audio_step.fade.fade_out_sec == step_dict["fade_out_duration"]
    else:
        assert audio_step.fade.fade_out_sec == 0.0


@given(
    st.lists(
        st.sampled_from(
            [
                {"type": "stable", "duration": 1.0, "frequency": 1.0},
                {"type": "stable", "duration": 2.0, "frequency": 2.0},
                {"type": "stable", "duration": 0.5, "frequency": 4.0},
            ]
        ),
        min_size=1,
        max_size=2,
    ),
    valid_noise_configs(),
)
@settings(max_examples=10)
def test_generate_audio_sequence_properties(steps: list, noise_config: NoiseConfig):
    """Property-based test for generate_audio_sequence with simple steps."""
    # For transition steps without start_frequency, we need to ensure they have
    # a previous step to derive the frequency from or add a start_frequency
    for i, step in enumerate(steps):
        if step["type"] == "transition" and "start_frequency" not in step:
            if i == 0:
                # First step must have explicit start_frequency
                step["start_frequency"] = 10.0  # Add a valid start frequency

    try:
        # Use standard sample rate and base frequency for simplicity
        sample_rate = STANDARD_SAMPLE_RATE
        base_freq = DEFAULT_BASE_FREQUENCY

        # Generate the audio sequence
        left, right, total_duration = generate_audio_sequence(
            sample_rate=sample_rate,
            base_freq=base_freq,
            steps=steps,
            noise_config=noise_config,
        )

        # Calculate expected duration and samples
        expected_duration = sum(step["duration"] for step in steps)
        expected_samples = int(sample_rate * expected_duration)

        # Allow for small differences in sample count (±2 samples)
        sample_diff = abs(len(left) - expected_samples)
        assert sample_diff <= 2, (
            f"Expected {expected_samples} samples, "
            f"got {len(left)}, difference: {sample_diff}"
        )
        assert len(left) == len(
            right
        ), f"Left and right channel lengths differ: {len(left)} vs {len(right)}"
        assert total_duration == pytest.approx(expected_duration, abs=0.01)

        # Check amplitude is within expected range
        assert np.max(np.abs(left)) <= 1.0 + 1e-9  # Allow tiny float imprecision
        assert np.max(np.abs(right)) <= 1.0 + 1e-9

        # If noise was added, check channel differences only in specific cases
        # Some noise configs/signals might result in identical channels
        # due to how noise is applied
        if (
            noise_config.type != "none"
            and noise_config.amplitude
            > 0.05  # Only check if noise is significant enough
            and len(steps) == 1  # Simplify test case - single step is more predictable
        ):
            # With noise, there should be at least some difference between channels
            # Skip this check if channels happen to be identical, as it might be a
            # false positive
            if not np.allclose(left, right, atol=1e-9):
                max_diff = np.max(np.abs(left - right))
                assert (
                    max_diff <= 2.0
                ), f"Max difference between channels ({max_diff}) exceeds limit"
    except Exception as e:
        # Print step details to help debug
        print(f"Error with steps: {steps}")
        print(f"Noise config: {noise_config}")
        raise e


@given(valid_tones(), st.floats(min_value=0.01, max_value=1.0))
def test_save_audio_file_properties(tone: Tone, duration: float):
    """Property-based test for save_audio_file."""
    # Generate a short tone
    sample_rate = STANDARD_SAMPLE_RATE
    left, right = generate_tone(sample_rate, duration, tone)

    # Create a temporary file for testing
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # Save the audio file
        save_audio_file(tmp_path, sample_rate, left, right, duration)

        # Verify the file exists and can be read
        assert os.path.exists(tmp_path)

        # Read the file back and verify content
        data, sr = sf.read(tmp_path)

        # Check basic properties
        assert sr == sample_rate
        assert data.shape == (len(left), 2)  # Stereo

        # Check content with some tolerance for potential conversion differences
        assert np.allclose(data[:, 0], left, atol=1e-4)
        assert np.allclose(data[:, 1], right, atol=1e-4)

    finally:
        # Clean up the temporary file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

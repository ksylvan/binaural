"""Unit tests for the tone generator module."""

import numpy as np
import pytest
import soundfile as sf

from binaural.data_types import NoiseConfig, Tone
from binaural.exceptions import (
    AudioGenerationError,
    ConfigurationError,
    UnsupportedFormatError,
)
from binaural.tone_generator import (
    config_step_to_audio_step,
    generate_audio_sequence,
    generate_tone,
    save_audio_file,
)

# Constants for tests
SAMPLE_RATE = 44100
BASE_FREQ = 100
DURATION_SEC = 1.0
NUM_SAMPLES = int(SAMPLE_RATE * DURATION_SEC)


# --- Tests for config_step_to_audio_step ---


def test_config_step_to_audio_step_stable():
    """Test converting a valid stable step config."""
    step_dict = {"type": "stable", "frequency": 10, "duration": 5}
    audio_step = config_step_to_audio_step(step_dict, None)
    assert audio_step.duration == 5
    assert audio_step.freq.type == "stable"
    assert audio_step.freq.start == 10
    assert audio_step.freq.end == 10
    assert audio_step.fade.fade_in_sec == 0.0
    assert audio_step.fade.fade_out_sec == 0.0


def test_config_step_to_audio_step_transition_explicit_start():
    """Test converting a valid transition step with explicit start frequency."""
    step_dict = {
        "type": "transition",
        "start_frequency": 10,
        "end_frequency": 5,
        "duration": 10,
        "fade_in_duration": 1,
        "fade_out_duration": 2,
    }
    audio_step = config_step_to_audio_step(step_dict, None)  # No previous freq needed
    assert audio_step.duration == 10
    assert audio_step.freq.type == "transition"
    assert audio_step.freq.start == 10
    assert audio_step.freq.end == 5
    assert audio_step.fade.fade_in_sec == 1
    assert audio_step.fade.fade_out_sec == 2


def test_config_step_to_audio_step_transition_implicit_start():
    """Test converting a valid transition step using previous frequency."""
    step_dict = {"type": "transition", "end_frequency": 5, "duration": 10}
    previous_freq = 8.0
    audio_step = config_step_to_audio_step(step_dict, previous_freq)
    assert audio_step.duration == 10
    assert audio_step.freq.type == "transition"
    assert audio_step.freq.start == previous_freq
    assert audio_step.freq.end == 5


def test_config_step_to_audio_step_missing_key():
    """Test error handling for missing keys in step config."""
    with pytest.raises(ConfigurationError, match="'type' key"):
        config_step_to_audio_step({"frequency": 10, "duration": 5}, None)
    with pytest.raises(ConfigurationError, match="'duration' key"):
        config_step_to_audio_step({"type": "stable", "frequency": 10}, None)
    with pytest.raises(ConfigurationError, match="'frequency' key"):
        config_step_to_audio_step({"type": "stable", "duration": 5}, None)
    with pytest.raises(ConfigurationError, match="'end_frequency'"):
        config_step_to_audio_step({"type": "transition", "duration": 5}, 10.0)


def test_config_step_to_audio_step_invalid_type():
    """Test error handling for invalid step type."""
    with pytest.raises(ConfigurationError, match="Invalid step type 'unknown'"):
        config_step_to_audio_step({"type": "unknown", "duration": 5}, None)


def test_config_step_to_audio_step_first_transition_no_start():
    """Test error when first transition step omits start_frequency."""
    step_dict = {"type": "transition", "end_frequency": 5, "duration": 10}
    with pytest.raises(
        ConfigurationError, match="must explicitly define 'start_frequency'"
    ):
        config_step_to_audio_step(step_dict, None)  # No previous frequency


# --- Tests for generate_tone ---


def test_generate_tone_stable():
    """Generate a stable tone and verify basic properties."""
    tone = Tone(
        base_freq=BASE_FREQ,
        freq_diff_start=5,
        freq_diff_end=5,
        fade_in_sec=0,
        fade_out_sec=0,
    )
    left, right = generate_tone(SAMPLE_RATE, DURATION_SEC, tone)
    assert len(left) == NUM_SAMPLES
    assert len(right) == NUM_SAMPLES
    assert np.max(np.abs(left)) <= 1.0
    assert np.max(np.abs(right)) <= 1.0


def test_generate_tone_transition():
    """Generate a transitioning tone."""
    tone = Tone(
        base_freq=BASE_FREQ,
        freq_diff_start=5,
        freq_diff_end=15,
        fade_in_sec=0.1,
        fade_out_sec=0.1,
    )
    left, right = generate_tone(SAMPLE_RATE, DURATION_SEC, tone)
    assert len(left) == NUM_SAMPLES
    assert len(right) == NUM_SAMPLES


def test_generate_tone_zero_duration():
    """Test generating a tone with zero duration."""
    tone = Tone(base_freq=BASE_FREQ, freq_diff_start=5, freq_diff_end=5)
    left, right = generate_tone(SAMPLE_RATE, 0.0, tone)
    assert len(left) == 0
    assert len(right) == 0


# --- Tests for generate_audio_sequence ---


def test_generate_audio_sequence_simple():
    """Test generating a sequence with one stable step."""
    steps = [{"type": "stable", "frequency": 10, "duration": DURATION_SEC}]
    noise_config = NoiseConfig()  # Default no noise
    left, right, total_duration = generate_audio_sequence(
        SAMPLE_RATE, BASE_FREQ, steps, noise_config
    )
    assert total_duration == DURATION_SEC
    assert len(left) == NUM_SAMPLES
    assert len(right) == NUM_SAMPLES


def test_generate_audio_sequence_multi_step():
    """Test generating a sequence with multiple steps (stable and transition)."""
    steps = [
        {"type": "stable", "frequency": 10, "duration": 0.5},
        {"type": "transition", "end_frequency": 5, "duration": 0.5},
    ]
    noise_config = NoiseConfig()
    left, right, total_duration = generate_audio_sequence(
        SAMPLE_RATE, BASE_FREQ, steps, noise_config
    )
    expected_duration = 0.5 + 0.5
    expected_samples = int(SAMPLE_RATE * expected_duration)
    assert total_duration == pytest.approx(expected_duration)
    assert len(left) == expected_samples
    assert len(right) == expected_samples


def test_generate_audio_sequence_with_noise():
    """Test generating audio sequence with background noise."""
    steps = [{"type": "stable", "frequency": 10, "duration": DURATION_SEC}]
    noise_amplitude = 0.2
    noise_config = NoiseConfig(type="white", amplitude=noise_amplitude)

    # Generate without noise first for comparison
    left_no_noise, _, _ = generate_audio_sequence(
        SAMPLE_RATE, BASE_FREQ, steps, NoiseConfig()
    )
    rms_no_noise = np.sqrt(np.mean(left_no_noise**2))

    # Generate with noise
    left_noise, right_noise, total_duration = generate_audio_sequence(
        SAMPLE_RATE, BASE_FREQ, steps, noise_config
    )
    rms_noise = np.sqrt(np.mean(left_noise**2))

    assert total_duration == DURATION_SEC
    assert len(left_noise) == NUM_SAMPLES
    assert len(right_noise) == NUM_SAMPLES

    assert not np.isclose(rms_noise, rms_no_noise, atol=1e-2)

    # Check that the max amplitude is still <= 1.0 due to scaling
    assert np.max(np.abs(left_noise)) <= 1.0 + 1e-9  # Allow for float precision
    assert np.max(np.abs(right_noise)) <= 1.0 + 1e-9


def test_generate_audio_sequence_empty_steps():
    """Test that generate_audio_sequence raises ConfigurationError for empty steps."""
    noise_config = NoiseConfig()
    with pytest.raises(ConfigurationError, match="No steps defined"):
        generate_audio_sequence(SAMPLE_RATE, BASE_FREQ, [], noise_config)


def test_generate_audio_sequence_invalid_step():
    """Test error handling for an invalid step within the sequence."""
    steps = [{"type": "stable"}]  # Missing frequency and duration
    noise_config = NoiseConfig()
    with pytest.raises(ConfigurationError, match="Error processing step 1"):
        generate_audio_sequence(SAMPLE_RATE, BASE_FREQ, steps, noise_config)


def test_save_audio_file_wav(tmp_path):
    """Generate dummy stereo audio and save to a temporary WAV file."""
    t = np.linspace(0, DURATION_SEC, NUM_SAMPLES, endpoint=False)
    left = np.sin(2 * np.pi * 100 * t) * 0.5
    right = np.sin(2 * np.pi * 105 * t) * 0.5
    file_path = tmp_path / "test.wav"

    save_audio_file(str(file_path), SAMPLE_RATE, left, right, DURATION_SEC)

    assert file_path.exists()
    # Verify content by reading back
    data, sr = sf.read(str(file_path))
    assert sr == SAMPLE_RATE
    assert data.shape == (NUM_SAMPLES, 2)
    assert np.allclose(data[:, 0], left, atol=1e-4)  # PCM_16 introduces quantization
    assert np.allclose(data[:, 1], right, atol=1e-4)


def test_save_audio_file_flac(tmp_path):
    """Generate dummy stereo audio and save to a temporary FLAC file."""
    t = np.linspace(0, DURATION_SEC, NUM_SAMPLES, endpoint=False)
    left = np.sin(2 * np.pi * 100 * t) * 0.5
    right = np.sin(2 * np.pi * 105 * t) * 0.5
    file_path = tmp_path / "test.flac"

    save_audio_file(str(file_path), SAMPLE_RATE, left, right, DURATION_SEC)

    assert file_path.exists()
    # Verify content by reading back
    data, sr = sf.read(str(file_path))
    assert sr == SAMPLE_RATE
    assert data.shape == (NUM_SAMPLES, 2)
    # FLAC is lossless for PCM_16, but conversion might have tiny diffs
    assert np.allclose(data[:, 0], left, atol=1e-4)
    assert np.allclose(data[:, 1], right, atol=1e-4)


def test_save_audio_file_unsupported_format(tmp_path):
    """Test saving to an unsupported format raises UnsupportedFormatError."""
    left = np.zeros(100)
    right = np.zeros(100)
    file_path = tmp_path / "test.mp3"
    with pytest.raises(UnsupportedFormatError, match="Unsupported format '.mp3'"):
        save_audio_file(str(file_path), SAMPLE_RATE, left, right, 1.0)


def test_save_audio_file_empty_data(tmp_path):
    """Test saving empty audio data raises AudioGenerationError."""
    left = np.array([])
    right = np.array([])
    file_path = tmp_path / "test.wav"
    with pytest.raises(AudioGenerationError, match="No audio data generated"):
        save_audio_file(str(file_path), SAMPLE_RATE, left, right, 0.0)


def test_save_audio_file_create_dir(tmp_path):
    """Test that save_audio_file creates the output directory if it doesn't exist."""
    t = np.linspace(0, DURATION_SEC, NUM_SAMPLES, endpoint=False)
    left = np.sin(2 * np.pi * 100 * t) * 0.5
    right = np.sin(2 * np.pi * 105 * t) * 0.5
    output_dir = tmp_path / "new_audio_dir"
    file_path = output_dir / "test.flac"

    assert not output_dir.exists()
    save_audio_file(str(file_path), SAMPLE_RATE, left, right, DURATION_SEC)

    assert output_dir.exists()

    assert output_dir.is_dir()

    assert file_path.exists()

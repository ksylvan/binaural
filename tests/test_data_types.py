"""Unit tests for binaural.data_types module."""

import pytest

from binaural_generator.core.data_types import AudioStep, FadeInfo, FrequencyRange


def test_audio_step_invalid_duration():
    """Test that creating an AudioStep with invalid duration raises ValueError."""
    fade = FadeInfo(fade_in_sec=3, fade_out_sec=3)
    freq = FrequencyRange(type="stable", start=10)
    with pytest.raises(ValueError):
        AudioStep(freq=freq, fade=fade, duration=5)


def test_frequency_range_invalid_type():
    """Test that creating a FrequencyRange with invalid type raises ValueError."""
    with pytest.raises(ValueError):
        FrequencyRange(type="invalid", start=10)

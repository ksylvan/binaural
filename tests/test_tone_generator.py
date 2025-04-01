"""Unit tests for the tone generator module."""

import os

import numpy as np
import pytest

from binaural.data_types import Tone
from binaural.tone_generator import (
    generate_audio_sequence,
    generate_tone,
    save_audio_file,
)


def test_generate_tone():
    "Generate a 1-second tone and verify array length matches sample_rate * duration"
    sample_rate = 44100
    duration = 1
    tone = Tone(
        base_freq=100, freq_diff_start=5, freq_diff_end=5, fade_in_sec=0, fade_out_sec=0
    )
    left, right = generate_tone(sample_rate, duration, tone)
    assert len(left) == sample_rate * duration
    assert len(right) == sample_rate * duration


def test_generate_audio_sequence_empty():
    "With an empty steps list, configuration error must be raised"
    sample_rate = 44100
    base_freq = 100
    with pytest.raises(Exception):
        generate_audio_sequence(sample_rate, base_freq, [])


def test_save_audio_file(tmp_path):
    "Generate dummy stereo audio and save to a temporary file"
    sample_rate = 44100
    duration = 1
    t = np.linspace(0, duration, sample_rate, endpoint=False)
    left = np.sin(2 * np.pi * 100 * t)
    right = np.sin(2 * np.pi * 105 * t)
    total_duration = duration
    file_path = tmp_path / "test.wav"
    save_audio_file(str(file_path), sample_rate, left, right, total_duration)
    assert os.path.exists(str(file_path))

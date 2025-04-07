"""Test the fade function in binaural module."""

import numpy as np

from binaural_generator.core.fade import apply_fade


def test_apply_fade_in_and_out():
    """Test that the apply_fade function applies fade in and fade out correctly."""
    sample_rate = 10
    duration = 2
    num_samples = sample_rate * duration
    audio = np.ones(num_samples)
    faded = apply_fade(audio, sample_rate, fade_in_sec=1, fade_out_sec=1)
    # Check that the first and last samples are faded (approximately 0)
    assert faded[0] < 0.1
    assert faded[-1] < 0.1

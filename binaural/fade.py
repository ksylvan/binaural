"""Apply a linear fade-in and fade-out to audio data."""

import numpy as np


def apply_fade(audio_data, sample_rate, fade_in_sec, fade_out_sec) -> np.ndarray:
    """Applies a linear fade-in and fade-out to the audio data."""
    num_samples = len(audio_data)
    envelope = np.ones(num_samples)

    fade_in_samples = min(num_samples, int(sample_rate * fade_in_sec))
    if fade_in_samples > 0:
        envelope[:fade_in_samples] = np.linspace(0, 1, fade_in_samples)

    fade_out_samples = min(
        num_samples - fade_in_samples, int(sample_rate * fade_out_sec)
    )
    if fade_out_samples > 0:
        envelope[-fade_out_samples:] = np.linspace(1, 0, fade_out_samples)

    return audio_data * envelope

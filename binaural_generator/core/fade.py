"""Apply a linear fade-in and fade-out to audio data."""

import numpy as np


def apply_fade(
    audio_data: np.ndarray,
    sample_rate: int,
    fade_in_sec: float,
    fade_out_sec: float,
) -> np.ndarray:
    """Applies a linear fade-in and fade-out envelope to the audio data.

    The fades are applied within the duration of the audio_data.
    The fade durations are clamped to the length of the audio data and
    potential overlaps (if fade_in_sec + fade_out_sec > total_duration)
    are handled by prioritizing the fade-in and truncating the fade-out
    if necessary.

    Args:
        audio_data: A 1D numpy array containing the audio samples.
        sample_rate: The sample rate of the audio in Hz.
        fade_in_sec: The duration of the linear fade-in in seconds.
        fade_out_sec: The duration of the linear fade-out in seconds.

    Returns:
        A 1D numpy array containing the audio data with the fade envelope applied.
    """
    # Get the total number of samples in the audio data
    num_samples = len(audio_data)
    if num_samples == 0:
        # Return immediately if the audio data is empty
        return audio_data

    # Initialize the envelope as an array of ones (no change)
    envelope = np.ones(num_samples, dtype=audio_data.dtype)

    # Calculate the number of samples for the fade-in
    # Ensure it doesn't exceed the total number of samples
    fade_in_samples = min(num_samples, int(sample_rate * fade_in_sec))
    if fade_in_samples > 0:
        # Apply linear ramp from 0 to 1 for the fade-in part
        envelope[:fade_in_samples] = np.linspace(0, 1, fade_in_samples)

    # Calculate the number of samples for the fade-out
    # Ensure it doesn't exceed the remaining samples after the fade-in
    # This handles potential overlaps: fade-in takes precedence
    fade_out_samples = min(
        num_samples - fade_in_samples, int(sample_rate * fade_out_sec)
    )
    if fade_out_samples > 0:
        # Apply linear ramp from 1 to 0 for the fade-out part
        envelope[-fade_out_samples:] = np.linspace(1, 0, fade_out_samples)

    # Multiply the original audio data by the calculated envelope
    return audio_data * envelope

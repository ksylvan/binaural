"""Unit tests for the binaural.noise module."""

import numpy as np
from scipy import signal

from binaural.noise import NoiseFactory

(
    generate_white_noise,
    generate_pink_noise,
    generate_brown_noise,
) = (
    NoiseFactory.get_strategy("white").generate,
    NoiseFactory.get_strategy("pink").generate,
    NoiseFactory.get_strategy("brown").generate,
)

# Define constants for tests
SAMPLE_RATE = 44100
DURATION = 1  # seconds
NUM_SAMPLES = SAMPLE_RATE * DURATION


def test_generate_white_noise():
    """Test white noise generation."""
    # Test generation with valid number of samples
    noise = generate_white_noise(NUM_SAMPLES)
    assert isinstance(noise, np.ndarray), "Output should be a numpy array"
    assert len(noise) == NUM_SAMPLES, "Output length should match num_samples"
    assert np.max(np.abs(noise)) <= 1.0, "Noise should be normalized to [-1, 1]"
    # White noise should have a roughly flat power spectrum (check mean is close to 0)
    assert np.abs(np.mean(noise)) < 0.1, "Mean should be close to zero"

    # Test generation with zero samples
    noise_zero = generate_white_noise(0)
    assert len(noise_zero) == 0, "Output should be empty for zero samples"

    # Test generation with negative samples
    # (should also be empty or raise error, current impl returns empty)
    noise_neg = generate_white_noise(-100)
    assert len(noise_neg) == 0, "Output should be empty for negative samples"


def test_generate_pink_noise():
    """Test pink noise generation."""
    # Test generation with valid number of samples
    noise = generate_pink_noise(NUM_SAMPLES)
    assert isinstance(noise, np.ndarray), "Output should be a numpy array"
    assert len(noise) == NUM_SAMPLES, "Output length should match num_samples"
    assert np.max(np.abs(noise)) <= 1.0, "Noise should be normalized to [-1, 1]"

    # Qualitative check: Pink noise power should decrease with frequency.
    # Perform FFT and check power spectrum slope (approx -10dB/decade or -3dB/octave)
    # This is a statistical property, so exact match isn't guaranteed.
    freqs, psd = signal.welch(noise, fs=SAMPLE_RATE, nperseg=1024)
    # Ignore DC component and very low frequencies where estimation is poor
    valid_indices = (freqs > 10) & (freqs < SAMPLE_RATE / 2 - 100)
    log_freqs = np.log10(freqs[valid_indices])
    log_psd = 10 * np.log10(psd[valid_indices])  # Power in dB
    # Fit a line: slope should be roughly -10 dB/decade for pink noise (power ~ 1/f)
    slope, _ = np.polyfit(log_freqs, log_psd, 1)
    assert (
        -15 < slope < -5
    ), f"Slope {slope:.2f} out of expected range for pink noise (-10 dB/decade)"

    # Test generation with zero samples
    noise_zero = generate_pink_noise(0)
    assert len(noise_zero) == 0, "Output should be empty for zero samples"

    # Test generation with negative samples
    noise_neg = generate_pink_noise(-100)
    assert len(noise_neg) == 0, "Output should be empty for negative samples"


def test_generate_brown_noise():
    """Test brown noise generation."""
    # Test generation with valid number of samples
    noise = generate_brown_noise(NUM_SAMPLES)
    assert isinstance(noise, np.ndarray), "Output should be a numpy array"
    assert len(noise) == NUM_SAMPLES, "Output length should match num_samples"
    assert np.max(np.abs(noise)) <= 1.0, "Noise should be normalized to [-1, 1]"
    assert (
        np.abs(np.mean(noise)) < 1e-9
    ), "Mean should be very close to zero after centering"

    # Qualitative check: Brown noise power should decrease steeply with frequency.
    # Perform FFT and check power spectrum slope (approx -20dB/decade or -6dB/octave)
    freqs, psd = signal.welch(noise, fs=SAMPLE_RATE, nperseg=1024)
    valid_indices = (freqs > 10) & (freqs < SAMPLE_RATE / 2 - 100)
    log_freqs = np.log10(freqs[valid_indices])
    log_psd = 10 * np.log10(psd[valid_indices])  # Power in dB
    # Fit a line: slope should be roughly -20 dB/decade for brown noise (power ~ 1/f^2)
    slope, _ = np.polyfit(log_freqs, log_psd, 1)
    assert (
        -25 < slope < -15
    ), f"Slope {slope:.2f} out of expected range for brown noise (-20 dB/decade)"

    # Test generation with zero samples
    noise_zero = generate_brown_noise(0)
    assert len(noise_zero) == 0, "Output should be empty for zero samples"

    # Test generation with negative samples
    noise_neg = generate_brown_noise(-100)
    assert len(noise_neg) == 0, "Output should be empty for negative samples"

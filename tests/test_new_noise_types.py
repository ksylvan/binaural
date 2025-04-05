"""Tests for the new noise generation strategies."""

import numpy as np

from binaural.noise import (BlueNoiseStrategy, GreyNoiseStrategy, NoiseFactory,
                            OceanNoiseStrategy, RainNoiseStrategy,
                            VioletNoiseStrategy)
from tests.test_common import (check_basic_noise_properties,
                               estimate_spectral_slope)

# Test sample parameters
NUM_SAMPLES = 2**14  # Power of 2 for efficient FFT
SAMPLE_RATE = 44100
FREQS = np.fft.rfftfreq(NUM_SAMPLES, 1 / SAMPLE_RATE)


class TestNewNoiseTypes:
    """Tests for the new noise generation strategies."""

    def test_blue_noise_generation(self):
        """Test basic properties of the blue noise generator."""
        noise_generator = BlueNoiseStrategy()
        noise = noise_generator.generate(NUM_SAMPLES)

        # Check basic properties
        check_basic_noise_properties(noise, NUM_SAMPLES)

        # Check spectral properties (slope should be approximately +1)
        slope = estimate_spectral_slope(noise, FREQS)
        assert 0.8 <= slope <= 1.2, f"Expected slope near 1, got {slope}"

    def test_violet_noise_generation(self):
        """Test basic properties of the violet noise generator."""
        noise_generator = VioletNoiseStrategy()
        noise = noise_generator.generate(NUM_SAMPLES)

        # Check basic properties
        check_basic_noise_properties(noise, NUM_SAMPLES)

        # Check spectral properties (slope should be approximately +2)
        slope = estimate_spectral_slope(noise, FREQS)
        assert 1.8 <= slope <= 2.2, f"Expected slope near 2, got {slope}"

    def test_grey_noise_generation(self):
        """Test basic properties of the grey noise generator."""
        noise_generator = GreyNoiseStrategy()
        noise = noise_generator.generate(NUM_SAMPLES)

        # Check basic properties
        check_basic_noise_properties(noise, NUM_SAMPLES)

        # Grey noise should have energy concentrated in mid frequencies (2-5 kHz)
        # We'll use a simple check to verify frequency content
        psd = np.abs(np.fft.rfft(noise))

        # Identify frequency bands
        low_band = FREQS < 1000
        mid_band = (FREQS >= 2000) & (FREQS <= 5000)
        high_band = FREQS > 10000

        # Calculate average energy in each band
        low_energy = np.mean(psd[low_band])
        mid_energy = np.mean(psd[mid_band])
        high_energy = np.mean(psd[high_band])

        # Check that mid-range frequencies have more energy than low or high
        assert mid_energy > low_energy, "Mid frequencies should be louder than low"
        assert mid_energy > high_energy, "Mid frequencies should be louder than high"

    def test_factory_initialization(self):
        """Test that the factory correctly initializes new noise types."""
        # Check that the new noise types are in the factory's strategies
        strategies = NoiseFactory.strategies()
        assert "blue" in strategies
        assert "violet" in strategies
        assert "grey" in strategies
        assert "rain" in strategies
        assert "ocean" in strategies # Check for ocean

        # Check that the factory returns the correct strategy instances
        blue_strategy = NoiseFactory.get_strategy("blue")
        violet_strategy = NoiseFactory.get_strategy("violet")
        grey_strategy = NoiseFactory.get_strategy("grey")
        rain_strategy = NoiseFactory.get_strategy("rain")
        ocean_strategy = NoiseFactory.get_strategy("ocean") # Get ocean

        assert isinstance(blue_strategy, BlueNoiseStrategy)
        assert isinstance(violet_strategy, VioletNoiseStrategy)
        assert isinstance(grey_strategy, GreyNoiseStrategy)
        assert isinstance(rain_strategy, RainNoiseStrategy)
        assert isinstance(ocean_strategy, OceanNoiseStrategy) # Check ocean type

    def test_empty_input(self):
        """Test that all strategies handle empty input correctly."""
        for noise_type in ["blue", "violet", "grey", "rain", "ocean"]:
            strategy = NoiseFactory.get_strategy(noise_type)
            result = strategy.generate(0)
            assert isinstance(result, np.ndarray)
            assert len(result) == 0

    def test_small_input(self):
        """Test with very small input sizes."""
        for noise_type in ["blue", "violet", "grey", "rain", "ocean"]:
            strategy = NoiseFactory.get_strategy(noise_type)
            result = strategy.generate(10)
            assert len(result) == 10
            # Allow slightly larger tolerance for combined noises like ocean/rain
            assert np.all(np.abs(result) <= 1.01)

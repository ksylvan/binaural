"""Tests for the rain noise generation strategy."""

import numpy as np

from binaural.noise import NoiseFactory, RainNoiseStrategy
from tests.test_common import check_basic_noise_properties, check_generate_output

# Test sample parameters
NUM_SAMPLES = 2**14  # Power of 2 for efficient FFT
SAMPLE_RATE = 44100
FREQS = np.fft.rfftfreq(NUM_SAMPLES, 1 / SAMPLE_RATE)


class TestRainNoiseStrategy:
    """Tests for the rain noise generation strategy."""

    def test_rain_noise_generation(self):
        """Test basic properties of the rain noise generator."""
        noise_generator = RainNoiseStrategy()
        noise = noise_generator.generate(NUM_SAMPLES)

        # Check basic properties
        check_basic_noise_properties(noise, NUM_SAMPLES)

        # Analyze frequency content
        psd = np.abs(np.fft.rfft(noise))

        # Identify frequency bands characteristic for rain sound
        very_low_band = FREQS < 100
        low_band = (FREQS >= 100) & (FREQS < 500)
        mid_high_band = (FREQS >= 2000) & (FREQS < 6000)
        very_high_band = FREQS >= 12000

        # Calculate average energy in each band
        very_low_energy = np.mean(psd[very_low_band])
        low_energy = np.mean(psd[low_band])
        mid_high_energy = np.mean(psd[mid_high_band])
        very_high_energy = np.mean(psd[very_high_band])

        # Rain noise should have energy primarily in mid to high frequencies
        # Very low frequencies should have the least energy
        assert (
            very_low_energy < low_energy
        ), "Very low frequencies should have less energy than low"

        # Mid-high frequencies should have more energy than very high (droplet impacts)
        assert (
            mid_high_energy > very_high_energy
        ), "Mid-high should have more energy than very high"

        # Check overall spectral balance matches expected rain profile
        # Mid frequencies should have significant energy compared to extremes
        assert (
            mid_high_energy > very_low_energy
        ), "Mid-high should have more energy than very low"

    def test_temporal_pattern(self):
        """Test that rain noise has appropriate temporal pattern."""
        # Use larger sample to detect temporal variations
        noise_generator = RainNoiseStrategy()
        long_sample = noise_generator.generate(SAMPLE_RATE * 2)  # 2 seconds

        # Rain noise should have amplitude variations due to droplet patterns
        # Split the sample into short segments and measure RMS
        segment_length = 1000  # ~23 ms segments to detect individual droplets
        num_segments = len(long_sample) // segment_length

        segment_energies = []
        for i in range(num_segments):
            start_idx = i * segment_length
            end_idx = start_idx + segment_length
            segment = long_sample[start_idx:end_idx]
            energy = np.sqrt(np.mean(segment**2))  # RMS energy
            segment_energies.append(energy)

        # Calculate the variation in segment energies
        # Rain should have noticeable variations as droplets hit
        energy_std = np.std(segment_energies)
        energy_mean = np.mean(segment_energies)
        variation_coefficient = energy_std / energy_mean

        # Rain should have a certain level of temporal variation
        # This is a relatively loose test since the exact parameters may change
        assert variation_coefficient > 0.05, "Rain should have temporal variations"

    def test_factory_integration(self):
        """Test that the rain noise type is correctly registered in the factory."""
        # Check that the rain noise type is in the factory's strategies
        strategies = NoiseFactory.strategies()
        assert "rain" in strategies

        # Check that the factory returns the correct strategy instance
        rain_strategy = NoiseFactory.get_strategy("rain")
        assert isinstance(rain_strategy, RainNoiseStrategy)

    def test_edge_cases(self):
        """Test that rain noise handles edge cases correctly."""
        check_generate_output(RainNoiseStrategy())

"""Tests for the ocean noise generation strategy."""

import numpy as np

from binaural_generator.core.noise import NoiseFactory, OceanNoiseStrategy
from tests.test_common import check_generate_output

# Test sample parameters
NUM_SAMPLES = 2**15  # Longer duration to capture wave patterns
SAMPLE_RATE = 44100
FREQS = np.fft.rfftfreq(NUM_SAMPLES, 1 / SAMPLE_RATE)


class TestOceanNoiseStrategy:
    """Tests for the ocean noise generation strategy."""

    def test_ocean_noise_generation(self):
        """Test basic properties of the ocean noise generator."""
        noise_generator = OceanNoiseStrategy()
        noise = noise_generator.generate(NUM_SAMPLES)

        # Check basic properties
        assert len(noise) == NUM_SAMPLES
        assert -1.01 <= np.min(noise) <= 0.01
        assert -0.01 <= np.max(noise) <= 1.01

        # Analyze frequency content
        psd = np.abs(np.fft.rfft(noise))

        # Identify frequency bands
        low_rumble_band = (FREQS >= 10) & (FREQS < 500)
        wave_crest_band = (FREQS >= 500) & (FREQS < 4000)
        high_freq_band = FREQS >= 8000

        # Calculate average energy in each band
        low_energy = np.mean(psd[low_rumble_band]) if np.any(low_rumble_band) else 0
        wave_energy = np.mean(psd[wave_crest_band]) if np.any(wave_crest_band) else 0
        high_energy = np.mean(psd[high_freq_band]) if np.any(high_freq_band) else 0

        # Ocean noise should have significant low-frequency rumble
        # and noticeable energy in the mid-frequencies for waves.
        # High frequencies should be less prominent than low/mid.
        assert low_energy > 0, "Should have low frequency rumble energy"
        assert wave_energy > 0, "Should have wave crest frequency energy"
        assert (
            low_energy > high_energy
        ), "Low rumble should be stronger than high frequencies"
        # Wave energy might be comparable or slightly less than rumble,
        # depending on parameters
        assert (
            wave_energy > high_energy
        ), "Wave crests should be stronger than high frequencies"

    def test_temporal_wave_pattern(self):
        """Test that ocean noise exhibits periodic wave crest patterns."""
        # Use a longer sample to observe multiple waves
        noise_generator = OceanNoiseStrategy()
        long_sample = noise_generator.generate(SAMPLE_RATE * 30)  # 30 seconds

        # Ocean noise should have significant amplitude variations
        # Use a rolling window RMS to detect wave crests
        window_size = int(0.5 * SAMPLE_RATE)  # 0.5 second window
        if len(long_sample) < window_size:
            # Skip test if sample is too short for meaningful analysis
            return

        # Calculate rolling RMS
        squared_signal = long_sample**2
        # Pad signal for convolution to maintain length
        padded_signal = np.pad(
            squared_signal, (window_size // 2, window_size // 2), mode="edge"
        )
        rolling_mean_sq = np.convolve(
            padded_signal, np.ones(window_size) / window_size, mode="valid"
        )
        rolling_rms = np.sqrt(rolling_mean_sq)

        # Check that there are peaks (wave crests) and troughs
        rms_min = np.min(rolling_rms)
        rms_max = np.max(rolling_rms)
        rms_mean = np.mean(rolling_rms)

        assert rms_max > rms_min, "RMS should vary over time"
        # Expect peaks to be significantly higher than the mean/troughs
        # The ratio depends heavily on parameters, but should be noticeable
        assert (
            rms_max > rms_mean * 1.2
        ), "Wave crests should be significantly louder than average"
        assert (
            rms_max > rms_min * 1.5
        ), "Peak RMS should be significantly higher than minimum RMS"

    def test_factory_integration(self):
        """Test that the ocean noise type is correctly registered in the factory."""
        # Check that the ocean noise type is in the factory's strategies
        strategies = NoiseFactory.strategies()
        assert "ocean" in strategies

        # Check that the factory returns the correct strategy instance
        ocean_strategy = NoiseFactory.get_strategy("ocean")
        assert isinstance(ocean_strategy, OceanNoiseStrategy)

    def test_edge_cases(self):
        """Test that ocean noise handles edge cases correctly."""
        check_generate_output(OceanNoiseStrategy())

"""Functions for generating different types of background noise"""

from abc import ABC, abstractmethod
from typing import Type

import numpy as np
from scipy.fft import fft, ifft

rng = np.random.default_rng()


class NoiseStrategy(ABC):
    """Abstract base class for noise generation strategies."""

    def name(self) -> str:
        """Return the name of the noise strategy (e.g., 'whitenoise')."""
        return self.__class__.__name__.replace("Strategy", "").lower()

    @abstractmethod
    def generate(self, num_samples: int) -> np.ndarray:
        """Generate noise samples.

        Args:
            num_samples: The number of samples to generate.

        Returns:
            A numpy array containing noise samples, normalized to [-1, 1].
        """


class WhiteNoiseStrategy(NoiseStrategy):
    """White noise generation strategy - equal energy across all frequencies."""

    def generate(self, num_samples: int) -> np.ndarray:
        """Generate white noise samples.

        Args:
            num_samples: The number of samples to generate.

        Returns:
            A numpy array containing white noise samples normalized to [-1, 1].
        """
        if num_samples <= 0:
            return np.array([])

        # Generate random samples from a standard normal distribution
        noise = rng.standard_normal(num_samples)

        # Normalize to range [-1, 1] by dividing by the maximum absolute value
        # Avoid division by zero if all samples happen to be zero (highly unlikely)
        max_abs_noise = np.max(np.abs(noise))
        if max_abs_noise > 1e-9:  # Use a small threshold instead of exact zero
            noise /= max_abs_noise
        return noise


class PinkNoiseStrategy(NoiseStrategy):
    """Pink noise generation strategy - energy decreases with frequency (1/f)."""

    # Threshold for switching to chunked generation to manage memory
    CHUNK_THRESHOLD = 1_048_576  # 2^20 samples
    # Chunk size for FFT processing (power of 2 is often efficient)
    CHUNK_SIZE = 524288  # 2^19 samples

    def generate(self, num_samples: int) -> np.ndarray:
        """Generate pink noise samples using the FFT filtering method.

        Pink noise has a power spectral density proportional to 1/f.
        For large sample counts (above CHUNK_THRESHOLD), uses a chunked approach
        to avoid potential memory issues with very large FFTs.

        Args:
            num_samples: The number of samples to generate.

        Returns:
            A numpy array containing pink noise samples normalized to [-1, 1].
        """
        # Return empty array if no samples are requested
        if num_samples <= 0:
            return np.array([])

        # Use chunked generation for very large number of samples
        if num_samples > self.CHUNK_THRESHOLD:
            return self._generate_chunked(num_samples)

        # Generate directly for smaller or moderate sample counts
        return self._generate_direct(num_samples)

    def _generate_direct(self, num_samples: int) -> np.ndarray:
        """Internal implementation of direct FFT-based pink noise generation."""
        # Generate initial white noise
        white_noise = rng.standard_normal(num_samples)

        # Compute the Fast Fourier Transform (FFT) of the white noise
        fft_white = fft(white_noise)

        # Get the corresponding frequencies for the FFT components
        frequencies = np.fft.fftfreq(num_samples)

        # Create the frequency scaling factor (1/sqrt(f) for pink noise power)
        # Initialize scaling factor array
        scaling = np.ones_like(frequencies, dtype=float)
        # Find indices of non-zero frequencies
        non_zero_freq_indices = frequencies != 0
        # Apply the 1/sqrt(|f|) scaling to non-zero frequencies
        # Use np.abs() for negative frequencies
        scaling[non_zero_freq_indices] = 1.0 / np.sqrt(
            np.abs(frequencies[non_zero_freq_indices])
        )
        # The DC component (frequencies == 0) scaling remains 1,
        # avoiding division by zero.

        # Apply the scaling to the FFT of the white noise
        fft_pink = fft_white * scaling

        # Compute the Inverse Fast Fourier Transform (IFFT)
        # Take the real part as the result should be a real-valued signal
        pink_noise = np.real(ifft(fft_pink))

        # Normalize the resulting pink noise to the range [-1, 1]
        max_abs_noise = np.max(np.abs(pink_noise))
        if max_abs_noise > 1e-9:
            pink_noise /= max_abs_noise
        return pink_noise

    def _generate_chunked(self, num_samples: int) -> np.ndarray:
        """Generate pink noise in chunks to avoid memory issues with large FFTs."""
        # Calculate the number of chunks required
        num_chunks = (num_samples + self.CHUNK_SIZE - 1) // self.CHUNK_SIZE

        # Initialize the result array
        result = np.zeros(num_samples)

        # Generate noise chunk by chunk
        for i in range(num_chunks):
            # Calculate start and end indices for the current chunk
            start_idx = i * self.CHUNK_SIZE
            end_idx = min(start_idx + self.CHUNK_SIZE, num_samples)
            chunk_length = end_idx - start_idx

            # Generate a pink noise chunk using the direct method
            chunk = self._generate_direct(chunk_length)
            # Assign the generated chunk to the corresponding part of the result array
            result[start_idx:end_idx] = chunk

        # Final normalization across the entire signal to ensure consistent amplitude
        max_abs_result = np.max(np.abs(result))
        if max_abs_result > 1e-9:
            result /= max_abs_result
        return result


class BrownNoiseStrategy(NoiseStrategy):
    """
    Brown noise (Brownian/Red noise) generation strategy.
    Energy decreases steeply with frequency (1/f^2).
    Generated by integrating white noise (random walk).
    """

    def generate(self, num_samples: int) -> np.ndarray:
        """Generate brown noise samples.

        Args:
            num_samples: The number of samples to generate.

        Returns:
            A numpy array containing brown noise samples normalized to [-1, 1].
        """
        # Return empty array if no samples requested
        if num_samples <= 0:
            return np.array([])

        # Generate white noise (increments of the random walk)
        white_noise = rng.standard_normal(num_samples)

        # Integrate white noise using cumulative sum to get brown noise
        brown_noise = np.cumsum(white_noise)

        # Center the noise around zero by subtracting the mean
        brown_noise -= np.mean(brown_noise)

        # Normalize to range [-1, 1]
        max_abs_noise = np.max(np.abs(brown_noise))
        if max_abs_noise > 1e-9:
            brown_noise /= max_abs_noise
        return brown_noise


class NullNoiseStrategy(NoiseStrategy):
    """Null object pattern implementation for when no noise is requested."""

    def generate(self, num_samples: int) -> np.ndarray:
        """Generate an array of zeros (representing no noise).

        Args:
            num_samples: The number of samples to generate.

        Returns:
            A numpy array containing zeros.
        """
        # Return an array of zeros with the specified length
        return np.zeros(num_samples)


class NoiseFactory:
    """Factory class for creating noise generator strategy instances."""

    # Dictionary mapping noise type strings to strategy classes
    _strategies: dict[str, Type[NoiseStrategy]] = {
        "white": WhiteNoiseStrategy,
        "pink": PinkNoiseStrategy,
        "brown": BrownNoiseStrategy,
        "none": NullNoiseStrategy,  # Include 'none' for the Null strategy
    }

    @staticmethod
    def strategies() -> list[str]:
        """Return a list of available noise strategy type names."""
        # Return the keys from the strategies dictionary as a list
        return list(NoiseFactory._strategies.keys())

    @classmethod
    def get_strategy(cls, noise_type: str) -> NoiseStrategy:
        """Get the appropriate noise strategy instance for the given type.

        Args:
            noise_type: The type of noise to generate (case-sensitive string,
                        e.g., "white", "pink", "brown", "none").

        Returns:
            A NoiseStrategy instance corresponding to the requested noise type.

        Raises:
            ValueError: If an unsupported noise type string is provided.
        """
        # Look up the strategy class in the dictionary
        strategy_class = cls._strategies.get(noise_type)

        # If the type is not found, raise an error
        if strategy_class is None:
            supported_types = ", ".join(cls.strategies())
            raise ValueError(
                f"Unsupported noise type: '{noise_type}'. "
                f"Supported types are: {supported_types}."
            )

        # Instantiate and return the appropriate strategy class
        return strategy_class()

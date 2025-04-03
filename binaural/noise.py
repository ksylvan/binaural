"""Functions for generating different types of background noise"""

from abc import ABC, abstractmethod
from typing import Type

import numpy as np
from scipy.fft import fft, ifft


class NoiseStrategy(ABC):
    """Abstract base class for noise generation strategies."""

    def name(self) -> str:
        """Return the name of the noise strategy."""
        return self.__class__.__name__.lower()

    @abstractmethod
    def generate(self, num_samples: int) -> np.ndarray:
        """Generate noise samples.

        Args:
            num_samples: The number of samples to generate.

        Returns:
            A numpy array containing noise samples.
        """


class WhiteNoiseStrategy(NoiseStrategy):
    """White noise generation strategy - equal energy across all frequencies."""

    def generate(self, num_samples: int) -> np.ndarray:
        """Generate white noise samples.

        Args:
            num_samples: The number of samples to generate.

        Returns:
            A numpy array containing white noise samples.
        """
        if num_samples <= 0:
            return np.array([])

        # Generate random samples from a standard normal distribution
        noise = np.random.randn(num_samples)

        # Normalize to range [-1, 1] assuming practical limits
        # Avoid division by zero if all samples are zero
        max_abs_noise = np.max(np.abs(noise))
        if max_abs_noise > 0:
            noise /= max_abs_noise
        return noise


class PinkNoiseStrategy(NoiseStrategy):
    """Pink noise generation strategy - energy decreases with frequency (1/f)."""

    def generate(self, num_samples: int) -> np.ndarray:
        """Generate pink noise samples using the FFT filtering method.

        Pink noise has a power spectral density proportional to 1/f.
        For large sample counts, uses a chunked approach to improve performance.

        Args:
            num_samples: The number of samples to generate.

        Returns:
            A numpy array containing pink noise samples.
        """
        if num_samples <= 0:
            return np.array([])

        # For large sample counts, use a chunked approach to avoid memory issues
        # Use a threshold (e.g., 2^20 samples) to decide when to chunk
        if num_samples > 1_048_576:
            return self._generate_chunked(num_samples)

        # Generate directly for smaller sample counts
        return self._generate_direct(num_samples)

    def _generate_direct(self, num_samples: int) -> np.ndarray:
        """Internal implementation of direct FFT-based pink noise generation."""
        # Generate white noise
        white_noise = np.random.randn(num_samples)

        # Compute FFT
        fft_white = fft(white_noise)

        # Create frequency scaling factor (1/sqrt(f))
        frequencies = np.fft.fftfreq(num_samples)

        # Avoid division by zero for the DC component (f=0)
        # Set DC component scaling to 1 (or 0, results differ slightly, 1 is common)
        scaling = np.ones_like(frequencies, dtype=float)
        non_zero_freq_indices = frequencies != 0
        scaling[non_zero_freq_indices] = 1.0 / np.sqrt(
            np.abs(frequencies[non_zero_freq_indices])
        )

        # Apply scaling to FFT
        fft_pink = fft_white * scaling

        # Compute inverse FFT
        pink_noise = np.real(ifft(fft_pink))

        # Normalize to range [-1, 1]
        # Avoid division by zero if all samples are zero
        max_abs_noise = np.max(np.abs(pink_noise))
        if max_abs_noise > 0:
            pink_noise /= max_abs_noise
        return pink_noise

    def _generate_chunked(self, num_samples: int) -> np.ndarray:
        """Generate pink noise in chunks to avoid memory issues with large FFTs."""
        # Define a reasonable chunk size (power of 2 often good for FFT)
        chunk_size = 524288  # 2^19
        # Calculate the number of chunks needed
        num_chunks = (num_samples + chunk_size - 1) // chunk_size

        # Initialize the result array
        result = np.zeros(num_samples)
        # Generate noise chunk by chunk
        for i in range(num_chunks):
            # Calculate start and end indices for the current chunk
            start_idx = i * chunk_size
            end_idx = min(start_idx + chunk_size, num_samples)
            chunk_length = end_idx - start_idx

            # Generate a pink noise chunk using the direct method
            chunk = self._generate_direct(chunk_length)
            # Assign the generated chunk to the result array
            result[start_idx:end_idx] = chunk

        # Final normalization to ensure consistent amplitude across the whole signal
        # Avoid division by zero if all samples are zero
        max_abs_result = np.max(np.abs(result))
        if max_abs_result > 0:
            result /= max_abs_result
        return result


class BrownNoiseStrategy(NoiseStrategy):
    """
    Brown noise generation strategy - energy decreases steeply with frequency (1/f^2).
    """

    def generate(self, num_samples: int) -> np.ndarray:
        """Generate brown noise (Brownian noise or 1/f^2 noise).

        Brown noise is generated by integrating white noise (random walk).

        Args:
            num_samples: The number of samples to generate.

        Returns:
            A numpy array containing brown noise samples.
        """
        # Return empty array if no samples requested
        if num_samples <= 0:
            return np.array([])

        # Generate white noise (steps of the random walk)
        white_noise = np.random.randn(num_samples)

        # Integrate white noise using cumulative sum
        brown_noise = np.cumsum(white_noise)

        # Normalize to range [-1, 1]
        brown_noise -= np.mean(brown_noise)  # Center around zero
        # Avoid division by zero if all samples are zero (e.g., after centering)
        max_abs_noise = np.max(np.abs(brown_noise))
        if max_abs_noise > 0:
            brown_noise /= max_abs_noise
        return brown_noise


class NullNoiseStrategy(NoiseStrategy):
    """Null object pattern implementation for when no noise is requested."""

    def generate(self, num_samples: int) -> np.ndarray:
        """Generate an array of zeros (no noise).

        Args:
            num_samples: The number of samples to generate.

        Returns:
            A numpy array containing zeros.
        """
        return np.zeros(num_samples)


class NoiseFactory:
    """Factory class for creating noise generators."""

    _strategies: dict[str, Type[NoiseStrategy]] = {
        "white": WhiteNoiseStrategy,
        "pink": PinkNoiseStrategy,
        "brown": BrownNoiseStrategy,
        "none": NullNoiseStrategy,
    }

    @staticmethod
    def strategies() -> list[str]:
        """Return the available noise strategies as a list of strings."""
        return NoiseFactory._strategies.keys()

    @classmethod
    def get_strategy(cls, noise_type: str) -> NoiseStrategy:
        """Get the appropriate noise strategy for the given type.

        Args:
            noise_type: The type of noise to generate
            ("white", "pink", "brown", or "none").

        Returns:
            A NoiseStrategy instance for the requested noise type.

        Raises:
            ValueError: If an unsupported noise type is requested.
        """
        if noise_type not in cls._strategies:
            supported_types = ", ".join(cls.strategies())
            raise ValueError(
                f"Unsupported noise type: '{noise_type}'. "
                f"Supported types are: {supported_types}."
            )

        # Instantiate and return the appropriate strategy class
        return cls._strategies[noise_type]()

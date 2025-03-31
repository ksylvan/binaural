"""Types used in the binaural module."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Tone:
    """Tone data."""

    base_freq: float
    freq_diff_start: float
    freq_diff_end: float
    fade_in_sec: float = 0.0
    fade_out_sec: float = 0.0


@dataclass
class FrequencyRange:
    """Frequency range data."""

    min: float
    max: float


@dataclass
class AudioStep:
    """Audio step data."""

    type: str
    duration: float
    frequency: Optional[float] = None
    _frequency: Optional[float] = field(default=None, init=False, repr=False)
    start_frequency: Optional[float] = None
    end_frequency: Optional[float] = None
    fade_in_duration: float = 0.0
    fade_out_duration: float = 0.0

    @property
    def frequency(self) -> Optional[float]:  # noqa: F811
        """Get the frequency for the step."""
        if self.type == "stable":
            return self._frequency
        return None

    @frequency.setter
    def frequency(self, value: Optional[float]) -> None:
        """Set the frequency for the step."""
        self._frequency = value

    def __post_init__(self) -> None:
        """Validate the step type and parameters."""
        if self.type not in ("stable", "transition"):
            raise ValueError(
                f"Invalid step type '{self.type}'. Must be 'stable' or 'transition'."
            )
        if self.duration <= 0:
            raise ValueError("Step duration must be a positive number in seconds.")
        if self.fade_in_duration < 0:
            raise ValueError(
                "fade_in_duration must be a non-negative number in seconds."
            )
        if self.fade_out_duration < 0:
            raise ValueError(
                "fade_out_duration must be a non-negative number in seconds."
            )
        if self.fade_in_duration + self.fade_out_duration > self.duration:
            raise ValueError(
                f"Sum of fade_in_duration ({self.fade_in_duration}s) and fade_out_duration "
                f"({self.fade_out_duration}s) cannot exceed step duration ({self.duration}s)."
            )
        if self.type == "stable":
            if not isinstance(self.frequency, (int, float)):
                raise ValueError("Stable step must specify a valid 'frequency'.")
            self.start_frequency = self.end_frequency = self.frequency
        else:
            if self.start_frequency and not isinstance(
                self.start_frequency, (int, float)
            ):
                raise ValueError(
                    "Transition step must specify a valid 'start_frequency'."
                )
            if self.end_frequency and not isinstance(self.end_frequency, (int, float)):
                raise ValueError(
                    "Transition step must specify a valid 'end_frequency'."
                )

    def __str__(self) -> str:
        """String representation of the AudioStep."""
        fade_info = ""
        if self.fade_in_duration > 0:
            fade_info += f", fade-in {self.fade_in_duration/60.0:.2f}min"
        if self.fade_out_duration > 0:
            fade_info += f", fade-out {self.fade_out_duration/60.0:.2f}min"

        return (
            f"{self.type}, {self.start_frequency}Hz -> {self.end_frequency}Hz, "
            f"duration {self.duration/60.0:.2f}min{fade_info}"
        )

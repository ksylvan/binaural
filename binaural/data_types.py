"""Types used in the binaural module."""

from dataclasses import dataclass
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

    type: str
    start: float
    end: Optional[float] = None

    def __post_init__(self) -> None:
        """Validate the frequency range type and parameters."""
        if self.type not in ("stable", "transition"):
            raise ValueError(
                f"Invalid frequency range type '{self.type}'. Must be 'stable' or 'transition'."
            )
        if self.type == "stable":
            if not isinstance(self.start, (int, float)):
                raise ValueError("Stable frequency must be a valid number.")
            self.end = self.start
        else:
            if not isinstance(self.start, (int, float)):
                raise ValueError("Transition frequency must be a valid number.")
        if not isinstance(self.end, (int, float)):
            raise ValueError("Transition frequency must be a valid number.")
        if self.start < 0:
            raise ValueError("Frequency start must be a non-negative number.")
        if self.end < 0:
            raise ValueError("Frequency end must be a non-negative number.")


@dataclass
class FadeInfo:
    """Fade information data."""

    fade_in_sec: float = 0.0
    fade_out_sec: float = 0.0

    def __post_init__(self) -> None:
        """Validate the fade information."""
        if self.fade_in_sec < 0:
            raise ValueError("fade_in_sec must be a non-negative number.")
        if self.fade_out_sec < 0:
            raise ValueError("fade_out_sec must be a non-negative number.")


@dataclass
class AudioStep:
    """Audio step data."""

    freq: FrequencyRange
    fade: FadeInfo
    duration: float

    def __post_init__(self) -> None:
        """Validate the step type and parameters."""
        if self.duration <= 0:
            raise ValueError("Step duration must be a positive number in seconds.")
        if self.fade.fade_in_sec + self.fade.fade_out_sec > self.duration:
            raise ValueError(
                f"Sum of fade-in ({self.fade.fade_in_sec}s) and fade-out "
                f"({self.fade.fade_out_sec}s) cannot exceed step duration ({self.duration}s)."
            )

    def __str__(self) -> str:
        """String representation of the AudioStep."""
        fade_info = ""
        if self.fade.fade_in_sec > 0:
            fade_info += f", fade-in {self.fade.fade_in_sec/60.0:.2f}min"
        if self.fade.fade_out_sec > 0:
            fade_info += f", fade-out {self.fade.fade_out_sec/60.0:.2f}min"

        return (
            f"{self.freq.type}, {self.freq.start}Hz -> {self.freq.end}Hz, "
            f"duration {self.duration/60.0:.2f}min{fade_info}"
        )

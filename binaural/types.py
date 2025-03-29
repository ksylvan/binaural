"""Types used in the binaural module."""

from dataclasses import dataclass


@dataclass
class Tone:
    """Tone data."""

    base_freq: float
    freq_diff_start: float
    freq_diff_end: float
    fade_in_sec: float = 0.0
    fade_out_sec: float = 0.0

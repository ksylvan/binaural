"""Utility functions for loading and validating YAML configuration files."""

import sys
from typing import Tuple

import yaml


def load_yaml_config(path: str) -> dict:
    """Loads and validates YAML configuration."""
    try:
        with open(path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
        if not isinstance(config, dict) or "steps" not in config:
            raise ValueError("YAML must have 'steps' key.")
        return config
    except FileNotFoundError:
        # Handle file not found error
        sys.exit(f"Error: Config file '{path}' not found.")
    except yaml.YAMLError as e:
        # Handle YAML parsing errors
        sys.exit(f"Error parsing YAML file '{path}': {e}")
    except ValueError as e:
        # Handle configuration validation errors
        sys.exit(f"Configuration error: {e}")


def validate_step(
    step: dict, previous_freq: float | None
) -> Tuple[str, float, float, float, float, float]:
    """Validates and extracts necessary fields from a step, including fades."""
    step_type = step.get("type")
    duration_sec = step.get("duration")

    if step_type not in ("stable", "transition"):
        raise ValueError(
            f"Invalid step type '{step_type}'. Must be 'stable' or 'transition'."
        )
    if not isinstance(duration_sec, (int, float)) or duration_sec <= 0:
        raise ValueError("Step duration must be a positive number in seconds.")

    fade_in_sec = step.get("fade_in_duration", 0.0)
    fade_out_sec = step.get("fade_out_duration", 0.0)

    if not isinstance(fade_in_sec, (int, float)) or fade_in_sec < 0:
        raise ValueError("fade_in_duration must be a non-negative number in seconds.")
    if not isinstance(fade_out_sec, (int, float)) or fade_out_sec < 0:
        raise ValueError("fade_out_duration must be a non-negative number in seconds.")

    if fade_in_sec + fade_out_sec > duration_sec:
        raise ValueError(
            f"Sum of fade_in_duration ({fade_in_sec}s) and fade_out_duration ({fade_out_sec}s) "
            f"cannot exceed step duration ({duration_sec}s)."
        )

    if step_type == "stable":
        freq = step.get("frequency")
        if not isinstance(freq, (int, float)):
            raise ValueError("Stable step must specify a valid 'frequency'.")
        return step_type, duration_sec, freq, freq, fade_in_sec, fade_out_sec

    start_freq = step.get("start_frequency", previous_freq)
    end_freq = step.get("end_frequency")

    if start_freq is None:
        raise ValueError(
            "Transition step must specify 'start_frequency' if it's the first step "
            "or if the previous step's frequency is unknown."
        )
    if not isinstance(end_freq, (int, float)):
        raise ValueError(
            "Transition step must specify a valid numeric 'end_frequency'."
        )
    if not isinstance(start_freq, (int, float)):
        raise ValueError(
            "Transition step must specify valid numeric 'start_frequency' or have a previous step."
        )

    return step_type, duration_sec, start_freq, end_freq, fade_in_sec, fade_out_sec

"""Utility functions for loading and validating YAML configuration files."""

import logging

import yaml

from binaural.data_types import NoiseConfig
from binaural.exceptions import (
    BinauralError,
    ConfigFileNotFoundError,
    ConfigurationError,
    YAMLParsingError,
)

logger = logging.getLogger(__name__)


def load_yaml_config(path: str) -> dict:
    """Loads and validates YAML configuration.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        A dictionary containing the parsed and validated configuration.
        Includes a 'noise_config' key holding a NoiseConfig object.

    Raises:
        ConfigFileNotFoundError: If the file is not found.
        YAMLParsingError: If YAML parsing fails.
        ConfigurationError: If configuration is invalid.
    """
    try:
        # Attempt to open and read the YAML file
        with open(path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)

        # Ensure the root of the YAML is a dictionary
        if not isinstance(config, dict):
            raise ConfigurationError("YAML configuration root must be a dictionary.")

        # Check for the mandatory 'steps' key
        required_keys = ["steps"]
        missing_keys = [key for key in required_keys if key not in config]
        if missing_keys:
            raise ConfigurationError(f"Missing required keys: {missing_keys}")

        # Validate the 'steps' list
        if not isinstance(config["steps"], list) or not config["steps"]:
            raise ConfigurationError("'steps' must be a non-empty list.")

        noise_settings = config.get("background_noise", {})
        if not isinstance(noise_settings, dict):
            raise ConfigurationError(
                "'background_noise' section must be a dictionary (key-value pairs)."
            )

        noise_type = noise_settings.get("type", "none")
        noise_amplitude = noise_settings.get("amplitude", 0.0)

        try:
            # Create and validate the NoiseConfig object
            noise_config_obj = NoiseConfig(
                type=noise_type, amplitude=float(noise_amplitude)
            )
        except (ValueError, TypeError) as e:
            # Catch errors during NoiseConfig creation
            # (e.g., invalid type, non-float amplitude)
            raise ConfigurationError(
                f"Invalid 'background_noise' configuration: {e}"
            ) from e

        # Add the validated NoiseConfig object back into the main config dictionary
        config["noise_config"] = noise_config_obj

        logger.debug("YAML configuration loaded and validated from %s", path)
        logger.debug("Noise configuration: %s", noise_config_obj)

        return config

    except FileNotFoundError as e:
        # Raise specific error if the config file doesn't exist
        raise ConfigFileNotFoundError(f"Config file '{path}' not found.") from e
    except yaml.YAMLError as e:
        # Raise specific error for YAML syntax issues
        raise YAMLParsingError(f"Error parsing YAML file '{path}': {e}") from e
    except ConfigurationError:  # Re-raise config errors
        raise
    except Exception as e:
        # Catch any other unexpected errors during loading/parsing
        raise BinauralError(
            f"An unexpected error occurred loading config '{path}': {e}"
        ) from e

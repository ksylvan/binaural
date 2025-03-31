"""Utility functions for loading and validating YAML configuration files."""

import logging
import yaml
from binaural.exceptions import (
    ConfigFileNotFoundError,
    YAMLParsingError,
    ConfigurationError,
)

logger = logging.getLogger(__name__)


def load_yaml_config(path: str) -> dict:
    """Loads and validates YAML configuration.

    Args:
        path (str): Path to the YAML file.

    Returns:
        dict: Parsed configuration dictionary.

    Raises:
        ConfigFileNotFoundError: If the file is not found.
        YAMLParsingError: If there is an error parsing the YAML.
        ConfigurationError: If the configuration is invalid.
    """
    try:
        with open(path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
        if not isinstance(config, dict) or "steps" not in config:
            raise ConfigurationError("YAML must have 'steps' key.")
        logger.debug("YAML configuration loaded from %s", path)
        return config
    except FileNotFoundError as e:
        raise ConfigFileNotFoundError(f"Config file '{path}' not found.") from e
    except yaml.YAMLError as e:
        raise YAMLParsingError(f"Error parsing YAML file '{path}': {e}") from e
    except ValueError as e:
        raise ConfigurationError(f"Configuration error: {e}") from e

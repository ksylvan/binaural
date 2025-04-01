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

    Raises:
        ConfigFileNotFoundError: If the file is not found.
        YAMLParsingError: If YAML parsing fails.
        ConfigurationError: If configuration is invalid.
    """
    try:
        with open(path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)

        if not isinstance(config, dict):
            raise ConfigurationError("YAML configuration must be a dictionary.")

        required_keys = ["steps"]
        missing_keys = [key for key in required_keys if key not in config]

        if missing_keys:
            raise ConfigurationError(f"Missing required keys: {missing_keys}")

        if not isinstance(config["steps"], list) or not config["steps"]:
            raise ConfigurationError("'steps' must be a non-empty list.")

        logger.debug("YAML configuration loaded and validated from %s", path)

        return config

    except FileNotFoundError as e:
        raise ConfigFileNotFoundError(f"Config file '{path}' not found.") from e
    except yaml.YAMLError as e:
        raise YAMLParsingError(f"Error parsing YAML file '{path}': {e}") from e
